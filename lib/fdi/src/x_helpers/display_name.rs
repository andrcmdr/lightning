use std::any::type_name;

use crate::ty::Ty;
use crate::{Method, Provider};

struct WithDisplayName<F> {
    display_name: String,
    method: F,
}

impl<F, P> Method<P> for WithDisplayName<F>
where
    F: Method<P>,
{
    type Output = F::Output;

    #[inline(always)]
    fn display_name(&self) -> Option<String> {
        Some(self.display_name.clone())
    }

    #[inline(always)]
    fn events(&self) -> Option<crate::Eventstore> {
        self.method.events()
    }

    #[inline(always)]
    fn dependencies() -> Vec<Ty> {
        F::dependencies()
    }

    #[inline(always)]
    fn call(self, registry: &Provider) -> Self::Output {
        self.method.call(registry)
    }
}

#[inline(always)]
pub fn with_display_name<F, P>(
    f: F,
    display_name: impl Into<String>,
) -> impl Method<P, Output = F::Output>
where
    F: Method<P>,
{
    WithDisplayName {
        display_name: display_name.into(),
        method: f,
    }
}

#[inline(always)]
pub fn map_display_name<F, P, N>(f: F, map: N) -> impl Method<P, Output = F::Output>
where
    F: Method<P>,
    N: FnOnce(String) -> String,
{
    let display_name = (map)(
        f.display_name()
            .unwrap_or_else(|| String::from(type_name::<F::Output>())),
    );

    WithDisplayName {
        display_name,
        method: f,
    }
}
