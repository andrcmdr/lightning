mod deserialize;
mod model;
mod serialize;

use std::collections::HashMap;

use anyhow::bail;
use bytes::Bytes;
use tract_onnx::prelude;
use tract_onnx::prelude::{Framework, InferenceFact, InferenceModel, RunnableModel};
use tract_onnx::tract_hir::infer::InferenceOp;

use crate::opts::{BorshVector, Encoding};
use crate::runtime::model::Info;

// Todo: let's improve this.
// Ideally we want every node to run onnx with runtime extensions.
// It might be worthwhile to use a custom build of onnx
// with statically-linked extensions.
const ORT_EXTENSIONS_LIB_PATH: &str = "~/.lightning/onnx/libortextensions.dylib";

pub struct Session {
    /// The Onnx Runtime Session.
    onnx: RunnableModel<InferenceFact, Box<dyn InferenceOp>, InferenceModel>,
    /// Encoding that will be used for the input and output throughout the session.
    encoding: Encoding,
}

impl Session {
    pub fn new(mut model: Bytes, encoding: Encoding) -> anyhow::Result<Self> {
        Ok(Self {
            onnx: tract_onnx::onnx()
                .model_for_read(&mut model)?
                .into_runnable()?,
            encoding,
        })
    }

    /// Runs model on the input.
    pub fn run(&self, input: Bytes) -> anyhow::Result<RunOutput> {
        if input.is_empty() {
            bail!("invalid input length");
        }

        // Process input and pass it to the model.
        let output = match self.encoding {
            Encoding::Borsh => {
                let session_input = deserialize::deserialize_borsh(input)?;
                let session_outputs = self.onnx.run(prelude::tvec!(session_input.into()))?;
                RunOutput::Borsh(serialize::borsh_serialize_outputs(session_outputs)?)
            },
            Encoding::SafeTensors => {
                let session_input = deserialize::deserialize_safetensors(input)?;
                let session_outputs = self.onnx.run(prelude::tvec!(session_input.into()))?;
                RunOutput::SafeTensors(serialize::safetensors_serialize_outputs(session_outputs)?)
            },
        };

        Ok(output)
    }
}

pub enum RunOutput {
    SafeTensors(Bytes),
    Borsh(HashMap<String, BorshVector>),
}

