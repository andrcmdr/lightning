use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::model;

#[derive(Debug, Serialize, Deserialize)]
#[repr(u8)]
pub enum Task {
    /// Run a model on an input.
    Run {
        /// Indicates whether input is a data set.
        ///
        /// If true, dataset will need to be fetched from origin so
        /// input will be interpreted as a [`crate::stream::Data`] object.
        /// Otherwise, input will be provided directly to the model to process.
        is_dataset: bool,
        /// Input for this run.
        input: Bytes,
        /// Model to run.
        model: model::Opts,
    },
    /// Train a model.
    Train {
        /// Number of epochs.
        epochs: u32,
        /// Uri for getting the models from origin.
        model_uri: String,
        /// Uri for getting the training data from origin.
        train_data_uri: String,
        /// Uri for getting the training labels from origin.
        train_label_uri: String,
        /// Uri for getting the validation data from origin.
        validation_data_uri: String,
        /// Uri for getting the validation labels from origin.
        validation_label_uri: String,
    },
}