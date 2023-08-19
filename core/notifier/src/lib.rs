use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use lightning_interfaces::{
    application::SyncQueryRunnerInterface,
    infu_collection::{c, Collection},
    notifier::{Notification, NotifierInterface},
    ApplicationInterface,
};
use tokio::{sync::mpsc, time::sleep};

pub struct Notifier<C: Collection> {
    query_runner: c![C::ApplicationInterface::SyncExecutor],
}

impl<C: Collection> Clone for Notifier<C> {
    fn clone(&self) -> Self {
        Self {
            query_runner: self.query_runner.clone(),
        }
    }
}

impl<C: Collection> Notifier<C> {
    fn get_until_epoch_end(&self) -> Duration {
        let epoch_info = self.query_runner.get_epoch_info();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let until_epoch_ends: u64 = (epoch_info.epoch_end as u128)
            .saturating_sub(now)
            .try_into()
            .unwrap();
        Duration::from_millis(until_epoch_ends)
    }
}

#[async_trait]
impl<C: Collection> NotifierInterface<C> for Notifier<C> {
    fn init(query_runner: c![C::ApplicationInterface::SyncExecutor]) -> Self {
        Self { query_runner }
    }

    fn notify_on_new_epoch(&self, tx: mpsc::Sender<Notification>) {
        let until_epoch_end = self.get_until_epoch_end();
        tokio::spawn(async move {
            sleep(until_epoch_end).await;
            tx.send(Notification::NewEpoch).await.unwrap();
        });
    }

    fn notify_before_epoch_change(&self, duration: Duration, tx: mpsc::Sender<Notification>) {
        let until_epoch_end = self.get_until_epoch_end();
        if until_epoch_end > duration {
            tokio::spawn(async move {
                sleep(until_epoch_end - duration).await;
                tx.send(Notification::BeforeEpochChange).await.unwrap();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use lightning_application::{
        app::Application,
        config::{Config, Mode},
        genesis::Genesis,
        query_runner::QueryRunner,
    };
    use lightning_interfaces::{
        application::{ApplicationInterface, ExecutionEngineSocket},
        infu_collection::Collection,
        partial,
    };

    use super::*;

    partial!(TestBinding {
        ApplicationInterface = Application<Self>;
        NotifierInterface = Notifier<Self>;
    });

    const EPSILON: f64 = 0.1;

    fn init_app(epoch_time: u64) -> (ExecutionEngineSocket, QueryRunner) {
        let mut genesis = Genesis::load().expect("Failed to load genesis from file.");
        let epoch_start = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        genesis.epoch_start = epoch_start;
        genesis.epoch_time = epoch_time;
        let config = Config {
            genesis: Some(genesis),
            mode: Mode::Test,
        };

        let app = Application::<TestBinding>::init(config).unwrap();

        (app.transaction_executor(), app.sync_query())
    }

    #[tokio::test]
    async fn test_on_new_epoch() {
        let (_, query_runner) = init_app(2000);

        let notifier = Notifier::<TestBinding>::init(query_runner);

        // Request to be notified when the epoch ends.
        let (tx, mut rx) = mpsc::channel(2048);
        let now = SystemTime::now();
        notifier.notify_on_new_epoch(tx);

        // The epoch time is 2 secs, the notification will be send when the epoch ends,
        // hence, the notification should arrive approx. 2 secs after the request was made.
        if let Notification::NewEpoch = rx.recv().await.unwrap() {
            let elapsed_time = now.elapsed().unwrap();
            assert!((elapsed_time.as_secs_f64() - 2.0).abs() < EPSILON);
        }
    }

    #[tokio::test]
    async fn test_before_epoch_change() {
        let (_, query_runner) = init_app(3000);

        let notifier = Notifier::<TestBinding>::init(query_runner);

        // Request to be notified 1 sec before the epoch ends.
        let (tx, mut rx) = mpsc::channel(2048);
        let now = SystemTime::now();
        notifier.notify_before_epoch_change(Duration::from_secs(1), tx);

        // The epoch time is 3 secs, the notification will be send 1 sec before the epoch ends,
        // hence, the notification should arrive approx. 2 secs after the request was made.
        if let Notification::BeforeEpochChange = rx.recv().await.unwrap() {
            let elapsed_time = now.elapsed().unwrap();
            assert!((elapsed_time.as_secs_f64() - 2.0).abs() < EPSILON);
        }
    }
}
