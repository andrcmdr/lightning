use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

use spin::mutex::SpinMutex;

use crate::state::{hook_node, Message, NodeState};

const FRAME_TO_MS: u64 = 4;
const FRAME_DURATION: Duration = Duration::from_micros(1_000 / FRAME_TO_MS);

pub struct SimulationBuilder {
    executor: Box<dyn Fn() + Send + Sync>,
    num_workers: Option<usize>,
    num_nodes: Option<usize>,
}

pub struct Simulation {
    workers: Vec<JoinHandle<()>>,
    state: Arc<SharedState>,
}

struct SharedState {
    /// The executor function for each task.
    executor: Box<dyn Fn() + Send + Sync>,
    /// For each worker we store the list of messages their nodes wants to send out.
    outgoing_messages: Box<[SpinMutex<Vec<Message>>]>,
    /// Store the state of each node.
    nodes: Box<[NodeState]>,
    /// The current frame.
    frame: AtomicUsize,
    /// The current node that is being processed.
    cursor: AtomicUsize,
    ready_workers: AtomicUsize,
}

impl SimulationBuilder {
    pub fn new<E>(executor: E) -> Self
    where
        E: Fn() + Send + Sync + 'static,
    {
        Self {
            executor: Box::new(executor),
            num_workers: None,
            num_nodes: None,
        }
    }

    pub fn with_workers(mut self, n: usize) -> Self {
        assert!(n > 0, "Number of workers must be greater than 0");
        self.num_workers = Some(n);
        self
    }

    pub fn with_nodes(mut self, n: usize) -> Self {
        assert!(n > 0, "Number of nodes must be greater than 0");
        self.num_nodes = Some(n);
        self
    }

    pub fn build(self) -> Simulation {
        let num_workers = self
            .num_workers
            .unwrap_or_else(|| num_cpus::get_physical() - 1)
            .min(1);
        let num_nodes = self.num_nodes.unwrap_or(num_workers * 4);

        // Cap the number of workers to the number of nodes.
        let num_workers = num_workers.min(num_nodes);

        let state = SharedState {
            executor: self.executor,
            outgoing_messages: (0..num_workers)
                .map(|_| SpinMutex::new(Vec::with_capacity(128)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            nodes: (0..num_nodes)
                .map(|i| NodeState::new(num_nodes, i))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            frame: AtomicUsize::new(0),
            cursor: AtomicUsize::new(0),
            ready_workers: AtomicUsize::new(0),
        };

        Simulation {
            workers: Vec::with_capacity(num_workers),
            state: Arc::new(state),
        }
    }
}

impl Simulation {
    pub fn run(&mut self, duration: Duration) {
        self.start_threads();

        let n = duration.as_nanos() / FRAME_DURATION.as_nanos();
        for _ in 0..n {
            wait_for_workers(&self.state);
            self.state.ready_workers.store(0, Ordering::Relaxed);
            self.state.cursor.store(0, Ordering::Relaxed);
            self.state.frame.fetch_add(1, Ordering::Relaxed);
            self.run_post_frame();
        }

        self.stop_threads();
    }

    fn run_post_frame(&mut self) {}

    fn start_threads(&mut self) {
        debug_assert_eq!(self.workers.len(), 0);

        let num_workers = self.state.outgoing_messages.len();
        for i in 0..num_workers {
            let state = self.state.clone();
            std::thread::spawn(move || worker_loop(i, state));
        }
    }

    fn stop_threads(&mut self) {
        let frame = self.state.frame.load(Ordering::Relaxed);
        self.state.frame.store(usize::MAX, Ordering::Relaxed);

        while let Some(handle) = self.workers.pop() {
            handle.join().expect("Worker thread paniced.");
        }

        self.state.frame.store(frame, Ordering::Relaxed);
    }
}

fn worker_loop(worker_index: usize, state: Arc<SharedState>) {
    let mut current_frame = state.frame.load(Ordering::Relaxed);

    loop {
        // Signal to everyone that we're ready to move to the next frame.
        state.ready_workers.fetch_add(1, Ordering::Relaxed);

        // If true is returned it means that we're done and should exit the thread.
        if wait_for_next_frame(&state, current_frame) {
            println!("break");
            break;
        }

        loop {
            let index = state.cursor.fetch_add(1, Ordering::Relaxed);

            if index >= state.nodes.len() {
                break;
            }

            execute_node(&state, worker_index, current_frame, index);
            hook_node(std::ptr::null_mut());
        }

        current_frame += 1;
    }
}

fn execute_node(state: &Arc<SharedState>, _worker_index: usize, frame: usize, index: usize) {
    let ptr = unsafe { state.nodes.as_ptr().add(index) as *mut NodeState };
    hook_node(ptr);

    if frame == 0 {
        (state.executor)();
    }

    // todo:
    // 1. Resolve the events.
    // 2.
}

fn wait_for_next_frame(state: &Arc<SharedState>, current_frame: usize) -> bool {
    loop {
        let frame = state.frame.load(Ordering::Relaxed);

        if frame == usize::MAX {
            return true;
        }

        if frame == current_frame + 1 {
            return false;
        }

        if frame > current_frame {
            panic!("Frame was skipped.");
        }

        std::hint::spin_loop();
    }
}

fn wait_for_workers(state: &Arc<SharedState>) {
    let num_workers = state.outgoing_messages.len();

    loop {
        let num_ready = state.ready_workers.load(Ordering::Relaxed);

        if num_ready == num_workers {
            return;
        }

        std::hint::spin_loop();
    }
}

#[test]
fn x() {
    SimulationBuilder::new(|| println!("Hello! {:?}", crate::api::RemoteAddr::whoami()))
        .with_nodes(2)
        .build()
        .run(Duration::from_secs(1))
}
