Help on EncDecRNNTBPEModel in module nemo.collections.asr.models.rnnt_bpe_models object:

class EncDecRNNTBPEModel(nemo.collections.asr.models.rnnt_models.EncDecRNNTModel, nemo.collections.asr.parts.mixins.mixins.ASRBPEMixin)
 |  EncDecRNNTBPEModel(cfg: omegaconf.dictconfig.DictConfig, trainer: lightning.pytorch.trainer.trainer.Trainer = None)
 |
 |  Base class for encoder decoder RNNT-based models with subword tokenization.
 |
 |  Method resolution order:
 |      EncDecRNNTBPEModel
 |      nemo.collections.asr.models.rnnt_models.EncDecRNNTModel
 |      nemo.collections.asr.models.asr_model.ASRModel
 |      nemo.core.classes.modelPT.ModelPT
 |      lightning.pytorch.core.module.LightningModule
 |      lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
 |      lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
 |      lightning.pytorch.core.hooks.ModelHooks
 |      lightning.pytorch.core.hooks.DataHooks
 |      lightning.pytorch.core.hooks.CheckpointHooks
 |      torch.nn.modules.module.Module
 |      nemo.core.classes.common.Model
 |      nemo.core.classes.common.Typing
 |      nemo.core.classes.common.Serialization
 |      nemo.core.classes.common.FileIO
 |      nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO
 |      nemo.collections.common.parts.optional_cuda_graphs.WithOptionalCudaGraphs
 |      nemo.collections.asr.parts.mixins.mixins.ASRModuleMixin
 |      nemo.collections.asr.parts.mixins.asr_adapter_mixins.ASRAdapterModelMixin
 |      nemo.core.classes.mixins.adapter_mixins.AdapterModelPTMixin
 |      nemo.core.classes.mixins.adapter_mixins.AdapterModuleMixin
 |      nemo.collections.asr.models.asr_model.ExportableEncDecModel
 |      nemo.core.classes.exportable.Exportable
 |      nemo.collections.asr.parts.mixins.transcription.ASRTranscriptionMixin
 |      nemo.collections.asr.parts.mixins.transcription.TranscriptionMixin
 |      nemo.collections.asr.parts.mixins.mixins.ASRBPEMixin
 |      abc.ABC
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(self, cfg: omegaconf.dictconfig.DictConfig, trainer: lightning.pytorch.trainer.trainer.Trainer = None)
 |      Base class from which all NeMo models should inherit
 |
 |      Args:
 |          cfg (DictConfig):  configuration object.
 |              The cfg object should have (optionally) the following sub-configs:
 |
 |              * train_ds - to instantiate training dataset
 |              * validation_ds - to instantiate validation dataset
 |              * test_ds - to instantiate testing dataset
 |              * optim - to instantiate optimizer with learning rate scheduler
 |
 |          trainer (Optional): Pytorch Lightning Trainer instance
 |
 |  change_decoding_strategy(self, decoding_cfg: omegaconf.dictconfig.DictConfig, verbose: bool = True)
 |      Changes decoding strategy used during RNNT decoding process.
 |
 |      Args:
 |          decoding_cfg: A config for the decoder, which is optional. If the decoding type
 |              needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
 |          verbose: A flag to enable/disable logging.
 |
 |  change_vocabulary(self, new_tokenizer_dir: Union[str, omegaconf.dictconfig.DictConfig], new_tokenizer_type: str, decoding_cfg: Optional[omegaconf.dictconfig.DictConfig] = None)
 |      Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning
 |      on from pre-trained model. This method changes only decoder and leaves encoder and pre-processing
 |      modules unchanged. For example, you would use it if you want to use pretrained encoder when fine-tuning
 |      on data in another language, or when you'd need model to learn capitalization, punctuation
 |      and/or special characters.
 |
 |      Args:
 |          new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer
 |              (if the tokenizer type is `agg`)
 |          new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
 |          decoding_cfg: A config for the decoder, which is optional. If the decoding type
 |              needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
 |
 |      Returns: None
 |
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |
 |  list_available_models() -> List[nemo.core.classes.common.PretrainedModelInfo]
 |      This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
 |
 |      Returns:
 |          List of available pre-trained models.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __abstractmethods__ = frozenset()
 |
 |  __annotations__ = {}
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.collections.asr.models.rnnt_models.EncDecRNNTModel:
 |
 |  extract_rnnt_loss_cfg(self, cfg: Optional[omegaconf.dictconfig.DictConfig])
 |      Helper method to extract the rnnt loss name, and potentially its kwargs
 |      to be passed.
 |
 |      Args:
 |          cfg: Should contain `loss_name` as a string which is resolved to a RNNT loss name.
 |              If the default should be used, then `default` can be used.
 |              Optionally, one can pass additional kwargs to the loss function. The subdict
 |              should have a keyname as follows : `{loss_name}_kwargs`.
 |
 |              Note that whichever loss_name is selected, that corresponding kwargs will be
 |              selected. For the "default" case, the "{resolved_default}_kwargs" will be used.
 |
 |      Examples:
 |          .. code-block:: yaml
 |
 |              loss_name: "default"
 |              warprnnt_numba_kwargs:
 |                  kwargs2: some_other_val
 |
 |      Returns:
 |          A tuple, the resolved loss name as well as its kwargs (if found).
 |
 |  forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None)
 |      Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
 |      and this method only performs the first step - forward of the acoustic model.
 |
 |      Please refer to the `training_step` in order to see the full `forward` step for training - which
 |      performs the forward of the acoustic model, the prediction network and then the joint network.
 |      Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.
 |
 |      Please refer to the `validation_step` in order to see the full `forward` step for inference - which
 |      performs the forward of the acoustic model, the prediction network and then the joint network.
 |      Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.
 |
 |      Args:
 |          input_signal: Tensor that represents a batch of raw audio signals,
 |              of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
 |              `self.sample_rate` number of floating point values.
 |          input_signal_length: Vector of length B, that contains the individual lengths of the audio
 |              sequences.
 |          processed_signal: Tensor that represents a batch of processed audio signals,
 |              of shape (B, D, T) that has undergone processing via some DALI preprocessor.
 |          processed_signal_length: Vector of length B, that contains the individual lengths of the
 |              processed audio sequences.
 |
 |      Returns:
 |          A tuple of 2 elements -
 |          1) The log probabilities tensor of shape [B, T, D].
 |          2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
 |
 |  list_export_subnets(self)
 |      Returns default set of subnet names exported for this model
 |      First goes the one receiving input (input_example)
 |
 |  multi_test_epoch_end(self, outputs, dataloader_idx: int = 0)
 |      Adds support for multiple test datasets. Should be overriden by subclass,
 |      so as to obtain appropriate logs for each of the dataloaders.
 |
 |      Args:
 |          outputs: Same as that provided by LightningModule.on_validation_epoch_end()
 |              for a single dataloader.
 |          dataloader_idx: int representing the index of the dataloader.
 |
 |      Returns:
 |          A dictionary of values, optionally containing a sub-dict `log`,
 |          such that the values in the log will be pre-pended by the dataloader prefix.
 |
 |  multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0)
 |      Adds support for multiple validation datasets. Should be overriden by subclass,
 |      so as to obtain appropriate logs for each of the dataloaders.
 |
 |      Args:
 |          outputs: Same as that provided by LightningModule.on_validation_epoch_end()
 |              for a single dataloader.
 |          dataloader_idx: int representing the index of the dataloader.
 |
 |      Returns:
 |          A dictionary of values, optionally containing a sub-dict `log`,
 |          such that the values in the log will be pre-pended by the dataloader prefix.
 |
 |  on_after_backward(self)
 |      zero-out the gradients which any of them is NAN or INF
 |
 |  predict_step(self, batch, batch_idx, dataloader_idx=0)
 |      Step function called during :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`. By default, it calls
 |      :meth:`~lightning.pytorch.core.LightningModule.forward`. Override to add any processing logic.
 |
 |      The :meth:`~lightning.pytorch.core.LightningModule.predict_step` is used
 |      to scale inference on multi-devices.
 |
 |      To prevent an OOM error, it is possible to use :class:`~lightning.pytorch.callbacks.BasePredictionWriter`
 |      callback to write the predictions to disk or database after each batch or on epoch end.
 |
 |      The :class:`~lightning.pytorch.callbacks.BasePredictionWriter` should be used while using a spawn
 |      based accelerator. This happens for ``Trainer(strategy="ddp_spawn")``
 |      or training on 8 TPU cores with ``Trainer(accelerator="tpu", devices=8)`` as predictions won't be returned.
 |
 |      Args:
 |          batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
 |          batch_idx: The index of this batch.
 |          dataloader_idx: The index of the dataloader that produced this batch.
 |              (only if multiple dataloaders used)
 |
 |      Return:
 |          Predicted output (optional).
 |
 |      Example ::
 |
 |          class MyModel(LightningModule):
 |
 |              def predict_step(self, batch, batch_idx, dataloader_idx=0):
 |                  return self(batch)
 |
 |          dm = ...
 |          model = MyModel()
 |          trainer = Trainer(accelerator="gpu", devices=2)
 |          predictions = trainer.predict(model, dm)
 |
 |  set_decoding_type_according_to_loss(self, decoding_cfg)
 |
 |  set_export_config(self, args)
 |      Sets/updates export_config dictionary
 |
 |  setup_optim_normalization(self)
 |      Helper method to setup normalization of certain parts of the model prior to the optimization step.
 |
 |      Supported pre-optimization normalizations are as follows:
 |
 |      .. code-block:: yaml
 |
 |          # Variation Noise injection
 |          model:
 |              variational_noise:
 |                  std: 0.0
 |                  start_step: 0
 |
 |          # Joint - Length normalization
 |          model:
 |              normalize_joint_txu: false
 |
 |          # Encoder Network - gradient normalization
 |          model:
 |              normalize_encoder_norm: false
 |
 |          # Decoder / Prediction Network - gradient normalization
 |          model:
 |              normalize_decoder_norm: false
 |
 |          # Joint - gradient normalization
 |          model:
 |              normalize_joint_norm: false
 |
 |  setup_test_data(self, test_data_config: Union[omegaconf.dictconfig.DictConfig, Dict, NoneType])
 |      Sets up the test data loader via a Dict-like object.
 |
 |      Args:
 |          test_data_config: A config that contains the information regarding construction
 |              of an ASR Training dataset.
 |
 |      Supported Datasets:
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
 |
 |  setup_training_data(self, train_data_config: Union[omegaconf.dictconfig.DictConfig, Dict, NoneType])
 |      Sets up the training data loader via a Dict-like object.
 |
 |      Args:
 |          train_data_config: A config that contains the information regarding construction
 |              of an ASR Training dataset.
 |
 |      Supported Datasets:
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
 |
 |  setup_validation_data(self, val_data_config: Union[omegaconf.dictconfig.DictConfig, Dict, NoneType])
 |      Sets up the validation data loader via a Dict-like object.
 |
 |      Args:
 |          val_data_config: A config that contains the information regarding construction
 |              of an ASR Training dataset.
 |
 |      Supported Datasets:
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
 |          -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
 |
 |  test_step(self, batch, batch_idx, dataloader_idx=0)
 |      Operates on a single batch of data from the test set. In this step you'd normally generate examples or
 |      calculate anything of interest such as accuracy.
 |
 |      Args:
 |          batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
 |          batch_idx: The index of this batch.
 |          dataloader_idx: The index of the dataloader that produced this batch.
 |              (only if multiple dataloaders used)
 |
 |      Return:
 |          - :class:`~torch.Tensor` - The loss tensor
 |          - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
 |          - ``None`` - Skip to the next batch.
 |
 |      .. code-block:: python
 |
 |          # if you have one test dataloader:
 |          def test_step(self, batch, batch_idx): ...
 |
 |
 |          # if you have multiple test dataloaders:
 |          def test_step(self, batch, batch_idx, dataloader_idx=0): ...
 |
 |      Examples::
 |
 |          # CASE 1: A single test dataset
 |          def test_step(self, batch, batch_idx):
 |              x, y = batch
 |
 |              # implement your own
 |              out = self(x)
 |              loss = self.loss(out, y)
 |
 |              # log 6 example images
 |              # or generated text... or whatever
 |              sample_imgs = x[:6]
 |              grid = torchvision.utils.make_grid(sample_imgs)
 |              self.logger.experiment.add_image('example_images', grid, 0)
 |
 |              # calculate acc
 |              labels_hat = torch.argmax(out, dim=1)
 |              test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
 |
 |              # log the outputs!
 |              self.log_dict({'test_loss': loss, 'test_acc': test_acc})
 |
 |      If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
 |      setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.
 |
 |      .. code-block:: python
 |
 |          # CASE 2: multiple test dataloaders
 |          def test_step(self, batch, batch_idx, dataloader_idx=0):
 |              # dataloader_idx tells you which dataset this is.
 |              ...
 |
 |      Note:
 |          If you don't need to test you don't need to implement this method.
 |
 |      Note:
 |          When the :meth:`test_step` is called, the model has been put in eval mode and
 |          PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
 |          to training mode and gradients are enabled.
 |
 |  training_step(self, batch, batch_nb)
 |      Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
 |      logger.
 |
 |      Args:
 |          batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
 |          batch_idx: The index of this batch.
 |          dataloader_idx: The index of the dataloader that produced this batch.
 |              (only if multiple dataloaders used)
 |
 |      Return:
 |          - :class:`~torch.Tensor` - The loss tensor
 |          - ``dict`` - A dictionary which can include any keys, but must include the key ``'loss'`` in the case of
 |            automatic optimization.
 |          - ``None`` - In automatic optimization, this will skip to the next batch (but is not supported for
 |            multi-GPU, TPU, or DeepSpeed). For manual optimization, this has no special meaning, as returning
 |            the loss is not required.
 |
 |      In this step you'd normally do the forward pass and calculate the loss for a batch.
 |      You can also do fancier things like multiple forward passes or something model specific.
 |
 |      Example::
 |
 |          def training_step(self, batch, batch_idx):
 |              x, y, z = batch
 |              out = self.encoder(x)
 |              loss = self.loss(out, x)
 |              return loss
 |
 |      To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:
 |
 |      .. code-block:: python
 |
 |          def __init__(self):
 |              super().__init__()
 |              self.automatic_optimization = False
 |
 |
 |          # Multiple optimizers (e.g.: GANs)
 |          def training_step(self, batch, batch_idx):
 |              opt1, opt2 = self.optimizers()
 |
 |              # do training_step with encoder
 |              ...
 |              opt1.step()
 |              # do training_step with decoder
 |              ...
 |              opt2.step()
 |
 |      Note:
 |          When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
 |          normalized by ``accumulate_grad_batches`` internally.
 |
 |  transcribe(self, audio: Union[str, List[str], numpy.ndarray, torch.utils.data.dataloader.DataLoader], batch_size: int = 4, return_hypotheses: bool = False, partial_hypothesis: Optional[List[ForwardRef('Hypothesis')]] = None, num_workers: int = 0, channel_selector: Union[int, Iterable[int], str, NoneType] = None, augmentor: omegaconf.dictconfig.DictConfig = None, verbose: bool = True, timestamps: Optional[bool] = None, override_config: Optional[nemo.collections.asr.parts.mixins.transcription.TranscribeConfig] = None) -> Union[List[str], List[nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis], Tuple[List[str]], Tuple[List[nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis]]]
 |      Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.
 |
 |      Args:
 |          audio: (a single or list) of paths to audio files or a np.ndarray/tensor audio array or path
 |              to a manifest file.
 |              Can also be a dataloader object that provides values that can be consumed by the model.
 |              Recommended length per file is between 5 and 25 seconds.                 But it is possible to pass a few hours long file if enough GPU memory is available.
 |          batch_size: (int) batch size to use during inference.                 Bigger will result in better throughput performance but would use more memory.
 |          return_hypotheses: (bool) Either return hypotheses or text
 |              With hypotheses can do some postprocessing like getting timestamp or rescoring
 |          partial_hypothesis: Optional[List['Hypothesis']] - A list of partial hypotheses to be used during rnnt
 |              decoding. This is useful for streaming rnnt decoding. If this is not None, then the length of this
 |              list should be equal to the length of the audio list.
 |          num_workers: (int) number of workers for DataLoader
 |          channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels
 |              from multi-channel audio. If set to `'average'`, it performs averaging across channels.
 |              Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
 |          augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
 |          verbose: (bool) whether to display tqdm progress bar
 |          timestamps: Optional(Bool): timestamps will be returned if set to True as part of hypothesis object
 |              (output.timestep['segment']/output.timestep['word']). Refer to `Hypothesis` class for more details.
 |              Default is None and would retain the previous state set by using self.change_decoding_strategy().
 |          override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
 |              **Note**: All other arguments in the function will be ignored if override_config is passed.
 |              You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.
 |
 |      Returns:
 |          Returns a tuple of 2 items -
 |          * A list of greedy transcript texts / Hypothesis
 |          * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
 |
 |  validation_pass(self, batch, batch_idx, dataloader_idx=0)
 |
 |  validation_step(self, batch, batch_idx, dataloader_idx=0)
 |      Operates on a single batch of data from the validation set. In this step you'd might generate examples or
 |      calculate anything of interest like accuracy.
 |
 |      Args:
 |          batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
 |          batch_idx: The index of this batch.
 |          dataloader_idx: The index of the dataloader that produced this batch.
 |              (only if multiple dataloaders used)
 |
 |      Return:
 |          - :class:`~torch.Tensor` - The loss tensor
 |          - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
 |          - ``None`` - Skip to the next batch.
 |
 |      .. code-block:: python
 |
 |          # if you have one val dataloader:
 |          def validation_step(self, batch, batch_idx): ...
 |
 |
 |          # if you have multiple val dataloaders:
 |          def validation_step(self, batch, batch_idx, dataloader_idx=0): ...
 |
 |      Examples::
 |
 |          # CASE 1: A single validation dataset
 |          def validation_step(self, batch, batch_idx):
 |              x, y = batch
 |
 |              # implement your own
 |              out = self(x)
 |              loss = self.loss(out, y)
 |
 |              # log 6 example images
 |              # or generated text... or whatever
 |              sample_imgs = x[:6]
 |              grid = torchvision.utils.make_grid(sample_imgs)
 |              self.logger.experiment.add_image('example_images', grid, 0)
 |
 |              # calculate acc
 |              labels_hat = torch.argmax(out, dim=1)
 |              val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
 |
 |              # log the outputs!
 |              self.log_dict({'val_loss': loss, 'val_acc': val_acc})
 |
 |      If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument. We recommend
 |      setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.
 |
 |      .. code-block:: python
 |
 |          # CASE 2: multiple validation dataloaders
 |          def validation_step(self, batch, batch_idx, dataloader_idx=0):
 |              # dataloader_idx tells you which dataset this is.
 |              ...
 |
 |      Note:
 |          If you don't need to validate you don't need to implement this method.
 |
 |      Note:
 |          When the :meth:`validation_step` is called, the model has been put in eval mode
 |          and PyTorch gradients have been disabled. At the end of validation,
 |          the model goes back to training mode and gradients are enabled.
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from nemo.collections.asr.models.rnnt_models.EncDecRNNTModel:
 |
 |  decoder_joint
 |
 |  input_types
 |      Define these to enable input neural type checks
 |
 |  output_types
 |      Define these to enable output neural type checks
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from nemo.collections.asr.models.rnnt_models.EncDecRNNTModel:
 |
 |  wer
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.collections.asr.models.asr_model.ASRModel:
 |
 |  add_auxiliary_losses(self, loss: torch.Tensor, reset_registry: bool = False) -> torch.Tensor
 |      Utility method to enable calculation of auxiliary losses for ASR training.
 |
 |      Args:
 |          loss: The output loss value prior to addition with auxiliary losses.
 |          reset_registry: Bool, whether to reset the AccessMixin registry after adding auxiliary losses.
 |
 |      Returns:
 |          Loss tensor used for back propagation.
 |
 |  disable_cuda_graphs(self) -> bool
 |      Disable (maybe temporary) CUDA graphs, return True if state changed
 |
 |  maybe_enable_cuda_graphs(self, force_reinit=False) -> bool
 |      Enable CUDA graphs if all conditions met, return True if state changed
 |
 |  on_predict_epoch_start(self) -> None
 |      For predicting, we enable CUDA graphs to speedup validation.
 |      Force re-enabling required due to issues with trainer and mixed precision.
 |      We do not need to disable CUDA graphs after predicting, since `predict` cannot be called in training loop.
 |      EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs
 |
 |  on_test_epoch_start(self) -> None
 |      For testing, we enable CUDA graphs to speedup validation.
 |      Force re-enabling required due to issues with trainer and mixed precision.
 |      We do not need to disable CUDA graphs after testing, since `test` cannot be called in training loop.
 |      EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs.
 |
 |  on_train_epoch_end(self) -> None
 |      After training, we can enable the decoder with CUDA graphs.
 |      EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs
 |
 |  on_train_epoch_start(self) -> None
 |      Decoder with CUDA graphs does not release memory, thus we disable it for training epoch.
 |      EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs
 |
 |  on_validation_epoch_end(self) -> Optional[dict[str, dict[str, torch.Tensor]]]
 |      After validation, we disable CUDA graphs, since `validation` can be called in training loop, and
 |      training will continue after validation
 |      EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs.
 |
 |  on_validation_epoch_start(self) -> None
 |      For validation, we enable CUDA graphs to speedup validation.
 |      Force re-enabling required due to issues with trainer and mixed precision.
 |      EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs.
 |
 |  setup_optimization_flags(self)
 |      Utility method that must be explicitly called by the subclass in order to support optional optimization flags.
 |      This method is the only valid place to access self.cfg prior to DDP training occurs.
 |
 |      The subclass may chose not to support this method, therefore all variables here must be checked via hasattr()
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from nemo.collections.asr.models.asr_model.ASRModel:
 |
 |  oomptimizer_schema
 |      Return a typing schema for optimal batch size calibration for various
 |      sequence lengths using OOMptimizer.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.modelPT.ModelPT:
 |
 |  configure_optimizers(self)
 |      Configure the optimizer and scheduler.
 |
 |  get_test_dataloader_prefix(self, dataloader_idx: 'int' = 0) -> 'str'
 |      Get the name of one or more data loaders, which will be prepended to all logs.
 |
 |      Args:
 |          dataloader_idx: Index of the data loader.
 |
 |      Returns:
 |          str name of the data loader at index provided.
 |
 |  get_validation_dataloader_prefix(self, dataloader_idx: 'int' = 0) -> 'str'
 |      Get the name of one or more data loaders, which will be prepended to all logs.
 |
 |      Args:
 |          dataloader_idx: Index of the data loader.
 |
 |      Returns:
 |          str name of the data loader at index provided.
 |
 |  has_artifacts(self) -> 'bool'
 |      Returns True if model has artifacts registered
 |
 |  has_native_or_submodules_artifacts(self) -> 'bool'
 |      Returns True if it has artifacts or any of the submodules have artifacts
 |
 |  has_nemo_submodules(self) -> 'bool'
 |      Returns True if it has any registered NeMo submodules
 |
 |  load_part_of_state_dict(self, state_dict, include, exclude, load_from_string=None)
 |      Load a part of the state dict into the model.
 |
 |  maybe_init_from_pretrained_checkpoint(self, cfg: 'OmegaConf', map_location: 'str' = 'cpu')
 |      Initializes a given model with the parameters obtained via specific config arguments.
 |      The state dict of the provided model will be updated with `strict=False` setting so as to prevent
 |      requirement of exact model parameters matching.
 |
 |      Initializations:
 |          init_from_nemo_model: Str path to a .nemo model in order to load state_dict from single nemo file;
 |          if loading from multiple files, pass in a dict where the values have the following fields:
 |
 |              path: Str path to .nemo model
 |
 |              include: Optional list of strings, at least one of which needs to be contained in parameter name
 |              to be loaded from this .nemo file. Default: everything is included.
 |
 |              exclude: Optional list of strings, which can be used to exclude any parameter containing one of
 |              these strings from being loaded from this .nemo file. Default: nothing is excluded.
 |
 |              hydra usage example:
 |
 |              init_from_nemo_model:
 |                  model0:
 |                      path:<path/to/model1>
 |                      include:["encoder"]
 |                  model1:
 |                      path:<path/to/model2>
 |                      include:["decoder"]
 |                      exclude:["embed"]
 |
 |          init_from_pretrained_model: Str name of a pretrained model checkpoint (obtained via cloud).
 |              The model will be downloaded (or a cached copy will be used), instantiated and then
 |              its state dict will be extracted. If loading from multiple models, you can pass in a dict
 |              with the same format as for init_from_nemo_model, except with "name" instead of "path"
 |
 |          init_from_ptl_ckpt: Str name of a Pytorch Lightning checkpoint file. It will be loaded and
 |              the state dict will extracted. If loading from multiple files, you can pass in a dict
 |              with the same format as for init_from_nemo_model.
 |
 |      Args:
 |          cfg: The config used to instantiate the model. It need only contain one of the above keys.
 |          map_location: str or torch.device() which represents where the intermediate state dict
 |              (from the pretrained model or checkpoint) will be loaded.
 |
 |  named_nemo_modules(self, prefix_name: 'str' = '', prefix_config: 'str' = '') -> "Iterator[Tuple[str, str, 'ModelPT']]"
 |      Returns an iterator over all NeMo submodules recursively, yielding
 |      tuples of (attribute path, path in config, submodule), starting from the core module
 |
 |      Args:
 |          prefix_name: prefix for the name path
 |          prefix_config: prefix for the path in config
 |
 |      Returns:
 |          Iterator over (attribute path, path in config, submodule), starting from (prefix, self)
 |
 |  on_fit_start(self) -> 'None'
 |      Register debug hooks.
 |
 |  on_predict_end(self)
 |      PyTorch Lightning hook:
 |      https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-test-end
 |
 |  on_test_end(self)
 |      PyTorch Lightning hook:
 |      https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-test-end
 |
 |  on_test_epoch_end(self) -> 'Optional[Dict[str, Dict[str, torch.Tensor]]]'
 |      Default DataLoader for Test set which automatically supports multiple data loaders
 |      via `multi_test_epoch_end`.
 |
 |      If multi dataset support is not required, override this method entirely in base class.
 |      In such a case, there is no need to implement `multi_test_epoch_end` either.
 |
 |      .. note::
 |          If more than one data loader exists, and they all provide `test_loss`,
 |          only the `test_loss` of the first data loader will be used by default.
 |          This default can be changed by passing the special key `test_dl_idx: int`
 |          inside the `test_ds` config.
 |
 |      Args:
 |          outputs: Single or nested list of tensor outputs from one or more data loaders.
 |
 |      Returns:
 |          A dictionary containing the union of all items from individual data_loaders,
 |          along with merged logs from all data loaders.
 |
 |  on_train_batch_end(self, outputs, batch: 'Any', batch_idx: 'int', unused: 'int' = 0) -> 'None'
 |      PyTorch Lightning hook:
 |      https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-end
 |      We use it here to enable nsys profiling.
 |
 |  on_train_batch_start(self, batch: 'Any', batch_idx: 'int', unused: 'int' = 0) -> 'Optional[int]'
 |      PyTorch Lightning hook:
 |      https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-start
 |      We use it here to enable profiling and dynamic freezing.
 |
 |  on_train_end(self)
 |      PyTorch Lightning hook:
 |      https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-end
 |      We use it here to cleanup the dynamic freezing config.
 |
 |  on_train_start(self)
 |      PyTorch Lightning hook:
 |      https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-start
 |      We use it here to copy the relevant config for dynamic freezing.
 |
 |  prepare_test(self, trainer: "'Trainer'") -> 'bool'
 |      Helper method to check whether the model can safely be tested
 |      on a dataset after training (or loading a checkpoint).
 |
 |      ::
 |
 |          trainer = Trainer()
 |          if model.prepare_test(trainer):
 |              trainer.test(model)
 |
 |      Returns:
 |          bool which declares the model safe to test. Provides warnings if it has to
 |          return False to guide the user.
 |
 |  propagate_model_guid(self)
 |      Propagates the model GUID to all submodules, recursively.
 |
 |  register_artifact(self, config_path: 'str', src: 'str', verify_src_exists: 'bool' = True)
 |      Register model artifacts with this function. These artifacts (files) will be included inside .nemo file
 |      when model.save_to("mymodel.nemo") is called.
 |
 |      How it works:
 |
 |      1. It always returns existing absolute path which can be used during Model constructor call
 |          EXCEPTION: src is None or "" in which case nothing will be done and src will be returned
 |      2. It will add (config_path, model_utils.ArtifactItem()) pair to self.artifacts
 |
 |          .. code-block::
 |
 |              If "src" is local existing path:
 |                  then it will be returned in absolute path form.
 |              elif "src" starts with "nemo_file:unique_artifact_name":
 |                  .nemo will be untarred to a temporary folder location and an actual existing path will be returned
 |              else:
 |                  an error will be raised.
 |
 |      WARNING: use .register_artifact calls in your models' constructors.
 |      The returned path is not guaranteed to exist after you have exited your model's constructor.
 |
 |      Args:
 |          config_path (str): Artifact key. Usually corresponds to the model config.
 |          src (str): Path to artifact.
 |          verify_src_exists (bool): If set to False, then the artifact is optional and register_artifact will return
 |                                    None even if src is not found. Defaults to True.
 |
 |      Returns:
 |          str: If src is not None or empty it always returns absolute path which is guaranteed to exist during model
 |               instance life
 |
 |  register_nemo_submodule(self, name: 'str', config_field: 'str', model: "'ModelPT'") -> 'None'
 |      Adds a NeMo model as a submodule. Submodule can be accessed via the `name` attribute on the parent NeMo model
 |      this submodule was registered on (`self`).
 |      In the saving process, the whole parent model (self) is held as a solid model with artifacts
 |      from the child submodule, the submodule config will be saved to the `config_field` of the parent model.
 |      This method is necessary to create a nested model, e.g.
 |
 |      .. code-block:: python
 |
 |          class ParentModel(ModelPT):
 |              def __init__(self, cfg, trainer=None):
 |                  super().__init__(cfg=cfg, trainer=trainer)
 |
 |                  # annotate type for autocompletion and type checking (optional)
 |                  self.child_model: Optional[ChildModel] = None
 |                  if cfg.get("child_model") is not None:
 |                      self.register_nemo_submodule(
 |                          name="child_model",
 |                          config_field="child_model",
 |                          model=ChildModel(self.cfg.child_model, trainer=trainer),
 |                      )
 |                  # ... other code
 |
 |      Args:
 |          name: name of the attribute for the submodule
 |          config_field: field in config, where submodule config should be saved
 |          model: NeMo model, instance of ModelPT
 |
 |  save_to(self, save_path: 'str')
 |      Saves model instance (weights and configuration) into .nemo file
 |       You can use "restore_from" method to fully restore instance from .nemo file.
 |
 |      .nemo file is an archive (tar.gz) with the following:
 |          model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for
 |                              model's constructor
 |          model_wights.ckpt - model checkpoint
 |
 |      Args:
 |          save_path: Path to .nemo file where model instance should be saved
 |
 |  set_trainer(self, trainer: 'Trainer')
 |      Set an instance of Trainer object.
 |
 |      Args:
 |          trainer: PyTorch Lightning Trainer object.
 |
 |  set_world_size(self, trainer: 'Trainer')
 |      Determines the world size from the PyTorch Lightning Trainer.
 |      And then updates AppState.
 |
 |      Args:
 |          trainer (Trainer): PyTorch Lightning Trainer object
 |
 |  setup(self, stage: 'Optional[str]' = None)
 |      Called at the beginning of fit, validate, test, or predict.
 |      This is called on every process when using DDP.
 |
 |      Args:
 |          stage: fit, validate, test or predict
 |
 |  setup_megatron_optimization(self, optim_config: 'Union[Dict[str, Any], DictConfig]')
 |      Setup mcore optimizer config.
 |
 |      Args:
 |          optim_config: Nemo optim args used to set up Mcore optimizer options.
 |
 |  setup_multiple_test_data(self, test_data_config: 'Union[DictConfig, Dict]')
 |      (Optionally) Setups data loader to be used in test, with support for multiple data loaders.
 |
 |      Args:
 |          test_data_layer_config: test data layer parameters.
 |
 |  setup_multiple_validation_data(self, val_data_config: 'Union[DictConfig, Dict]')
 |      (Optionally) Setups data loader to be used in validation, with support for multiple data loaders.
 |
 |      Args:
 |          val_data_layer_config: validation data layer parameters.
 |
 |  setup_optimization(self, optim_config: 'Optional[Union[DictConfig, Dict]]' = None, optim_kwargs: 'Optional[Dict[str, Any]]' = None)
 |      Prepares an optimizer from a string name and its optional config parameters.
 |
 |      Args:
 |          optim_config: A dictionary containing the following keys:
 |
 |              * "lr": mandatory key for learning rate. Will raise ValueError if not provided.
 |              * "optimizer": string name pointing to one of the available optimizers in the registry.                 If not provided, defaults to "adam".
 |              * "opt_args": Optional list of strings, in the format "arg_name=arg_value".                 The list of "arg_value" will be parsed and a dictionary of optimizer kwargs                 will be built and supplied to instantiate the optimizer.
 |
 |          optim_kwargs: A dictionary with additional kwargs for the
 |              optimizer. Used for non-primitive types that are not
 |              compatible with OmegaConf.
 |
 |  setup_optimizer_param_groups(self)
 |      Used to create param groups for the optimizer.
 |      As an example, this can be used to specify per-layer learning rates:
 |
 |      optim.SGD([
 |                  {'params': model.base.parameters()},
 |                  {'params': model.classifier.parameters(), 'lr': 1e-3}
 |                  ], lr=1e-2, momentum=0.9)
 |
 |      See https://pytorch.org/docs/stable/optim.html for more information.
 |      By default, ModelPT will use self.parameters().
 |      Override this method to add custom param groups.
 |      In the config file, add 'optim_param_groups' to support different LRs
 |      for different components (unspecified params will use the default LR):
 |
 |      model:
 |          optim_param_groups:
 |              encoder:
 |                  lr: 1e-4
 |                  momentum: 0.8
 |              decoder:
 |                  lr: 1e-3
 |          optim:
 |              lr: 3e-3
 |              momentum: 0.9
 |
 |  summarize(self, max_depth: 'int' = 1) -> 'model_summary.ModelSummary'
 |      Summarize this LightningModule.
 |
 |      Args:
 |          max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
 |              layer summary off. Default: 1.
 |
 |      Return:
 |          The model summary object
 |
 |  teardown(self, stage: 'str')
 |      Called at the end of fit and test.
 |
 |      Args:
 |          stage: either 'fit' or 'test'
 |
 |  test_dataloader(self)
 |      Get the test dataloader.
 |
 |  train_dataloader(self)
 |      Get the training dataloader.
 |
 |  val_dataloader(self)
 |      Get the validation dataloader.
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from nemo.core.classes.modelPT.ModelPT:
 |
 |  __init_subclass__() -> 'None'
 |      This method is called when a class is subclassed.
 |
 |      The default implementation does nothing. It may be
 |      overridden to extend subclasses.
 |
 |  extract_state_dict_from(restore_path: 'str', save_dir: 'str', split_by_module: 'bool' = False, save_restore_connector: 'SaveRestoreConnector' = None)
 |      Extract the state dict(s) from a provided .nemo tarfile and save it to a directory.
 |
 |      Args:
 |          restore_path: path to .nemo file from which state dict(s) should be extracted
 |          save_dir: directory in which the saved state dict(s) should be stored
 |          split_by_module: bool flag, which determins whether the output checkpoint should
 |              be for the entire Model, or the individual module's that comprise the Model
 |          save_restore_connector (SaveRestoreConnector): Can be overrided to add custom save and restore logic.
 |
 |      Example:
 |          To convert the .nemo tarfile into a single Model level PyTorch checkpoint
 |          ::
 |          state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.nemo', './asr_ckpts')
 |
 |
 |          To restore a model from a Model level checkpoint
 |          ::
 |          model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
 |          model.load_state_dict(torch.load("./asr_ckpts/model_weights.ckpt"))
 |
 |
 |          To convert the .nemo tarfile into multiple Module level PyTorch checkpoints
 |          ::
 |          state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from(
 |                          'asr.nemo',
 |                          './asr_ckpts',
 |                          split_by_module=True
 |                      )
 |
 |
 |          To restore a module from a Module level checkpoint
 |          ::
 |          model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
 |
 |          # load the individual components
 |          model.preprocessor.load_state_dict(torch.load("./asr_ckpts/preprocessor.ckpt"))
 |          model.encoder.load_state_dict(torch.load("./asr_ckpts/encoder.ckpt"))
 |          model.decoder.load_state_dict(torch.load("./asr_ckpts/decoder.ckpt"))
 |
 |
 |      Returns:
 |          The state dict that was loaded from the original .nemo checkpoint
 |
 |  load_from_checkpoint(checkpoint_path: 'str', *args, map_location: 'Optional[Union[Dict[str, str], str, torch.device, int, Callable]]' = None, hparams_file: 'Optional[str]' = None, strict: 'bool' = True, **kwargs)
 |      Loads ModelPT from checkpoint, with some maintenance of restoration.
 |      For documentation, please refer to LightningModule.load_from_checkpoint() documentation.
 |
 |  restore_from(restore_path: 'str', override_config_path: 'Optional[Union[OmegaConf, str]]' = None, map_location: 'Optional[torch.device]' = None, strict: 'bool' = True, return_config: 'bool' = False, save_restore_connector: 'SaveRestoreConnector' = None, trainer: 'Optional[Trainer]' = None, validate_access_integrity: 'bool' = True)
 |      Restores model instance (weights and configuration) from .nemo file.
 |
 |      Args:
 |          restore_path: path to .nemo file from which model should be instantiated
 |          override_config_path: path to a yaml config that will override the internal
 |              config file or an OmegaConf / DictConfig object representing the model config.
 |          map_location: Optional torch.device() to map the instantiated model to a device.
 |              By default (None), it will select a GPU if available, falling back to CPU otherwise.
 |          strict: Passed to load_state_dict. By default True.
 |          return_config: If set to true, will return just the underlying config of the restored
 |              model as an OmegaConf DictConfig object without instantiating the model.
 |          trainer: Optional, a pytorch lightning Trainer object that will be forwarded to the
 |              instantiated model's constructor.
 |          save_restore_connector (SaveRestoreConnector): Can be overridden to add custom save and restore logic.
 |
 |          Example:
 |              ```
 |              model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
 |              assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
 |              ```
 |
 |      Returns:
 |          An instance of type cls or its underlying config (if return_config is set).
 |
 |  update_save_restore_connector(save_restore_connector)
 |      Update the save_restore_connector for the model.
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from nemo.core.classes.modelPT.ModelPT:
 |
 |  hparams
 |      Overwrite default hparams property to return the lastest model config.
 |      Without this change, the hparams property would return the old config if there was a direct change to
 |      self._cfg (e.g., in self.setup_optimization()) that was not done via `self.cfg = new_cfg`.
 |
 |  num_weights
 |      Utility property that returns the total number of parameters of the Model.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from nemo.core.classes.modelPT.ModelPT:
 |
 |  cfg
 |      Property that holds the finalized internal config of the model.
 |
 |      Note:
 |          Changes to this config are not reflected in the state of the model.
 |          Please create a new model using an updated config to properly update the model.
 |
 |  test_step_outputs
 |      Cached outputs of test_step. It can be a list of items (for single data loader) or a list of
 |      lists (for multiple data loaders).
 |
 |      Returns:
 |          List of outputs of test_step.
 |
 |  trainer
 |      Get the trainer object.
 |
 |  validation_step_outputs
 |      Cached outputs of validation_step. It can be a list of items (for single data loader) or a list of lists
 |      (for multiple data loaders).
 |
 |      Returns:
 |          List of outputs of validation_step.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from lightning.pytorch.core.module.LightningModule:
 |
 |  __getstate__(self) -> Dict[str, Any]
 |      Helper for pickle.
 |
 |  all_gather(self, data: Union[torch.Tensor, Dict, List, Tuple], group: Optional[Any] = None, sync_grads: bool = False) -> Union[torch.Tensor, Dict, List, Tuple]
 |      Gather tensors or collections of tensors from multiple processes.
 |
 |      This method needs to be called on all processes and the tensors need to have the same shape across all
 |      processes, otherwise your program will stall forever.
 |
 |      Args:
 |          data: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof.
 |          group: the process group to gather results from. Defaults to all processes (world)
 |          sync_grads: flag that allows users to synchronize gradients for the all_gather operation
 |
 |      Return:
 |          A tensor of shape (world_size, batch, ...), or if the input was a collection
 |          the output will also be a collection with tensors of this shape. For the special case where
 |          world_size is 1, no additional dimension is added to the tensor(s).
 |
 |  backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None
 |      Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
 |      implementation if you need to.
 |
 |      Args:
 |          loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
 |              holds the normalized value (scaled by 1 / accumulation steps).
 |
 |      Example::
 |
 |          def backward(self, loss):
 |              loss.backward()
 |
 |  clip_gradients(self, optimizer: torch.optim.optimizer.Optimizer, gradient_clip_val: Union[int, float, NoneType] = None, gradient_clip_algorithm: Optional[str] = None) -> None
 |      Handles gradient clipping internally.
 |
 |      Note:
 |          - Do not override this method. If you want to customize gradient clipping, consider using
 |            :meth:`configure_gradient_clipping` method.
 |          - For manual optimization (``self.automatic_optimization = False``), if you want to use
 |            gradient clipping, consider calling
 |            ``self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")``
 |            manually in the training step.
 |
 |      Args:
 |          optimizer: Current optimizer being used.
 |          gradient_clip_val: The value at which to clip gradients.
 |          gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
 |              to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm.
 |
 |  configure_callbacks(self) -> Union[Sequence[lightning.pytorch.callbacks.callback.Callback], lightning.pytorch.callbacks.callback.Callback]
 |      Configure model-specific callbacks. When the model gets attached, e.g., when ``.fit()`` or ``.test()`` gets
 |      called, the list or a callback returned here will be merged with the list of callbacks passed to the Trainer's
 |      ``callbacks`` argument. If a callback returned here has the same type as one or several callbacks already
 |      present in the Trainer's callbacks list, it will take priority and replace them. In addition, Lightning will
 |      make sure :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callbacks run last.
 |
 |      Return:
 |          A callback or a list of callbacks which will extend the list of callbacks in the Trainer.
 |
 |      Example::
 |
 |          def configure_callbacks(self):
 |              early_stop = EarlyStopping(monitor="val_acc", mode="max")
 |              checkpoint = ModelCheckpoint(monitor="val_loss")
 |              return [early_stop, checkpoint]
 |
 |  configure_gradient_clipping(self, optimizer: torch.optim.optimizer.Optimizer, gradient_clip_val: Union[int, float, NoneType] = None, gradient_clip_algorithm: Optional[str] = None) -> None
 |      Perform gradient clipping for the optimizer parameters. Called before :meth:`optimizer_step`.
 |
 |      Args:
 |          optimizer: Current optimizer being used.
 |          gradient_clip_val: The value at which to clip gradients. By default, value passed in Trainer
 |              will be available here.
 |          gradient_clip_algorithm: The gradient clipping algorithm to use. By default, value
 |              passed in Trainer will be available here.
 |
 |      Example::
 |
 |          def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
 |              # Implement your own custom logic to clip gradients
 |              # You can call `self.clip_gradients` with your settings:
 |              self.clip_gradients(
 |                  optimizer,
 |                  gradient_clip_val=gradient_clip_val,
 |                  gradient_clip_algorithm=gradient_clip_algorithm
 |              )
 |
 |  freeze(self) -> None
 |      Freeze all params for inference.
 |
 |      Example::
 |
 |          model = MyLightningModule(...)
 |          model.freeze()
 |
 |  log(self, name: str, value: Union[torchmetrics.metric.Metric, torch.Tensor, int, float], prog_bar: bool = False, logger: Optional[bool] = None, on_step: Optional[bool] = None, on_epoch: Optional[bool] = None, reduce_fx: Union[str, Callable] = 'mean', enable_graph: bool = False, sync_dist: bool = False, sync_dist_group: Optional[Any] = None, add_dataloader_idx: bool = True, batch_size: Optional[int] = None, metric_attribute: Optional[str] = None, rank_zero_only: bool = False) -> None
 |      Log a key, value pair.
 |
 |      Example::
 |
 |          self.log('train_loss', loss)
 |
 |      The default behavior per hook is documented here: :ref:`extensions/logging:Automatic Logging`.
 |
 |      Args:
 |          name: key to log. Must be identical across all processes if using DDP or any other distributed strategy.
 |          value: value to log. Can be a ``float``, ``Tensor``, or a ``Metric``.
 |          prog_bar: if ``True`` logs to the progress bar.
 |          logger: if ``True`` logs to the logger.
 |          on_step: if ``True`` logs at this step. The default value is determined by the hook.
 |              See :ref:`extensions/logging:Automatic Logging` for details.
 |          on_epoch: if ``True`` logs epoch accumulated metrics. The default value is determined by the hook.
 |              See :ref:`extensions/logging:Automatic Logging` for details.
 |          reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
 |          enable_graph: if ``True``, will not auto detach the graph.
 |          sync_dist: if ``True``, reduces the metric across devices. Use with care as this may lead to a significant
 |              communication overhead.
 |          sync_dist_group: the DDP group to sync across.
 |          add_dataloader_idx: if ``True``, appends the index of the current dataloader to
 |              the name (when using multiple dataloaders). If False, user needs to give unique names for
 |              each dataloader to not mix the values.
 |          batch_size: Current batch_size. This will be directly inferred from the loaded batch,
 |              but for some data structures you might need to explicitly provide it.
 |          metric_attribute: To restore the metric state, Lightning requires the reference of the
 |              :class:`torchmetrics.Metric` in your model. This is found automatically if it is a model attribute.
 |          rank_zero_only: Tells Lightning if you are calling ``self.log`` from every process (default) or only from
 |              rank 0. If ``True``, you won't be able to use this metric as a monitor in callbacks
 |              (e.g., early stopping). Warning: Improper use can lead to deadlocks! See
 |              :ref:`Advanced Logging <visualize/logging_advanced:rank_zero_only>` for more details.
 |
 |  log_dict(self, dictionary: Union[Mapping[str, Union[torchmetrics.metric.Metric, torch.Tensor, int, float]], torchmetrics.collections.MetricCollection], prog_bar: bool = False, logger: Optional[bool] = None, on_step: Optional[bool] = None, on_epoch: Optional[bool] = None, reduce_fx: Union[str, Callable] = 'mean', enable_graph: bool = False, sync_dist: bool = False, sync_dist_group: Optional[Any] = None, add_dataloader_idx: bool = True, batch_size: Optional[int] = None, rank_zero_only: bool = False) -> None
 |      Log a dictionary of values at once.
 |
 |      Example::
 |
 |          values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
 |          self.log_dict(values)
 |
 |      Args:
 |          dictionary: key value pairs.
 |              Keys must be identical across all processes if using DDP or any other distributed strategy.
 |              The values can be a ``float``, ``Tensor``, ``Metric``, or ``MetricCollection``.
 |          prog_bar: if ``True`` logs to the progress base.
 |          logger: if ``True`` logs to the logger.
 |          on_step: if ``True`` logs at this step.
 |              ``None`` auto-logs for training_step but not validation/test_step.
 |              The default value is determined by the hook.
 |              See :ref:`extensions/logging:Automatic Logging` for details.
 |          on_epoch: if ``True`` logs epoch accumulated metrics.
 |              ``None`` auto-logs for val/test step but not ``training_step``.
 |              The default value is determined by the hook.
 |              See :ref:`extensions/logging:Automatic Logging` for details.
 |          reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
 |          enable_graph: if ``True``, will not auto-detach the graph
 |          sync_dist: if ``True``, reduces the metric across GPUs/TPUs. Use with care as this may lead to a significant
 |              communication overhead.
 |          sync_dist_group: the ddp group to sync across.
 |          add_dataloader_idx: if ``True``, appends the index of the current dataloader to
 |              the name (when using multiple). If ``False``, user needs to give unique names for
 |              each dataloader to not mix values.
 |          batch_size: Current batch size. This will be directly inferred from the loaded batch,
 |              but some data structures might need to explicitly provide it.
 |          rank_zero_only: Tells Lightning if you are calling ``self.log`` from every process (default) or only from
 |              rank 0. If ``True``, you won't be able to use this metric as a monitor in callbacks
 |              (e.g., early stopping). Warning: Improper use can lead to deadlocks! See
 |              :ref:`Advanced Logging <visualize/logging_advanced:rank_zero_only>` for more details.
 |
 |  lr_scheduler_step(self, scheduler: Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau], metric: Optional[Any]) -> None
 |      Override this method to adjust the default way the :class:`~lightning.pytorch.trainer.trainer.Trainer` calls
 |      each scheduler. By default, Lightning calls ``step()`` and as shown in the example for each scheduler based on
 |      its ``interval``.
 |
 |      Args:
 |          scheduler: Learning rate scheduler.
 |          metric: Value of the monitor used for schedulers like ``ReduceLROnPlateau``.
 |
 |      Examples::
 |
 |          # DEFAULT
 |          def lr_scheduler_step(self, scheduler, metric):
 |              if metric is None:
 |                  scheduler.step()
 |              else:
 |                  scheduler.step(metric)
 |
 |          # Alternative way to update schedulers if it requires an epoch value
 |          def lr_scheduler_step(self, scheduler, metric):
 |              scheduler.step(epoch=self.current_epoch)
 |
 |  lr_schedulers(self) -> Union[NoneType, List[Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]], torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]
 |      Returns the learning rate scheduler(s) that are being used during training. Useful for manual optimization.
 |
 |      Returns:
 |          A single scheduler, or a list of schedulers in case multiple ones are present, or ``None`` if no
 |          schedulers were returned in :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`.
 |
 |  manual_backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None
 |      Call this directly from your :meth:`training_step` when doing optimizations manually. By using this,
 |      Lightning can ensure that all the proper scaling gets applied when using mixed precision.
 |
 |      See :ref:`manual optimization<common/optimization:Manual optimization>` for more examples.
 |
 |      Example::
 |
 |          def training_step(...):
 |              opt = self.optimizers()
 |              loss = ...
 |              opt.zero_grad()
 |              # automatically applies scaling, etc...
 |              self.manual_backward(loss)
 |              opt.step()
 |
 |      Args:
 |          loss: The tensor on which to compute gradients. Must have a graph attached.
 |          *args: Additional positional arguments to be forwarded to :meth:`~torch.Tensor.backward`
 |          **kwargs: Additional keyword arguments to be forwarded to :meth:`~torch.Tensor.backward`
 |
 |  optimizer_step(self, epoch: int, batch_idx: int, optimizer: Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer], optimizer_closure: Optional[Callable[[], Any]] = None) -> None
 |      Override this method to adjust the default way the :class:`~lightning.pytorch.trainer.trainer.Trainer` calls
 |      the optimizer.
 |
 |      By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example.
 |      This method (and ``zero_grad()``) won't be called during the accumulation phase when
 |      ``Trainer(accumulate_grad_batches != 1)``. Overriding this hook has no benefit with manual optimization.
 |
 |      Args:
 |          epoch: Current epoch
 |          batch_idx: Index of current batch
 |          optimizer: A PyTorch optimizer
 |          optimizer_closure: The optimizer closure. This closure must be executed as it includes the
 |              calls to ``training_step()``, ``optimizer.zero_grad()``, and ``backward()``.
 |
 |      Examples::
 |
 |          def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
 |              # Add your custom logic to run directly before `optimizer.step()`
 |
 |              optimizer.step(closure=optimizer_closure)
 |
 |              # Add your custom logic to run directly after `optimizer.step()`
 |
 |  optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: torch.optim.optimizer.Optimizer) -> None
 |      Override this method to change the default behaviour of ``optimizer.zero_grad()``.
 |
 |      Args:
 |          epoch: Current epoch
 |          batch_idx: Index of current batch
 |          optimizer: A PyTorch optimizer
 |
 |      Examples::
 |
 |          # DEFAULT
 |          def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
 |              optimizer.zero_grad()
 |
 |          # Set gradients to `None` instead of zero to improve performance (not required on `torch>=2.0.0`).
 |          def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
 |              optimizer.zero_grad(set_to_none=True)
 |
 |      See :meth:`torch.optim.Optimizer.zero_grad` for the explanation of the above example.
 |
 |  optimizers(self, use_pl_optimizer: bool = True) -> Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer, lightning.fabric.wrappers._FabricOptimizer, List[torch.optim.optimizer.Optimizer], List[lightning.pytorch.core.optimizer.LightningOptimizer], List[lightning.fabric.wrappers._FabricOptimizer]]
 |      Returns the optimizer(s) that are being used during training. Useful for manual optimization.
 |
 |      Args:
 |          use_pl_optimizer: If ``True``, will wrap the optimizer(s) in a
 |              :class:`~lightning.pytorch.core.optimizer.LightningOptimizer` for automatic handling of precision,
 |              profiling, and counting of step calls for proper logging and checkpointing. It specifically wraps the
 |              ``step`` method and custom optimizers that don't have this method are not supported.
 |
 |      Returns:
 |          A single optimizer, or a list of optimizers in case multiple ones are present.
 |
 |  print(self, *args: Any, **kwargs: Any) -> None
 |      Prints only from process 0. Use this in any distributed mode to log only once.
 |
 |      Args:
 |          *args: The thing to print. The same as for Python's built-in print function.
 |          **kwargs: The same as for Python's built-in print function.
 |
 |      Example::
 |
 |          def forward(self, x):
 |              self.print(x, 'in forward')
 |
 |  to_onnx(self, file_path: Union[str, pathlib.Path], input_sample: Optional[Any] = None, **kwargs: Any) -> None
 |      Saves the model in ONNX format.
 |
 |      Args:
 |          file_path: The path of the file the onnx model should be saved to.
 |          input_sample: An input for tracing. Default: None (Use self.example_input_array)
 |          **kwargs: Will be passed to torch.onnx.export function.
 |
 |      Example::
 |
 |          class SimpleModel(LightningModule):
 |              def __init__(self):
 |                  super().__init__()
 |                  self.l1 = torch.nn.Linear(in_features=64, out_features=4)
 |
 |              def forward(self, x):
 |                  return torch.relu(self.l1(x.view(x.size(0), -1)
 |
 |          model = SimpleModel()
 |          input_sample = torch.randn(1, 64)
 |          model.to_onnx("export.onnx", input_sample, export_params=True)
 |
 |  to_torchscript(self, file_path: Union[str, pathlib.Path, NoneType] = None, method: Optional[str] = 'script', example_inputs: Optional[Any] = None, **kwargs: Any) -> Union[torch.ScriptModule, Dict[str, torch.ScriptModule]]
 |      By default compiles the whole model to a :class:`~torch.jit.ScriptModule`. If you want to use tracing,
 |      please provided the argument ``method='trace'`` and make sure that either the `example_inputs` argument is
 |      provided, or the model has :attr:`example_input_array` set. If you would like to customize the modules that are
 |      scripted you should override this method. In case you want to return multiple modules, we recommend using a
 |      dictionary.
 |
 |      Args:
 |          file_path: Path where to save the torchscript. Default: None (no file saved).
 |          method: Whether to use TorchScript's script or trace method. Default: 'script'
 |          example_inputs: An input to be used to do tracing when method is set to 'trace'.
 |            Default: None (uses :attr:`example_input_array`)
 |          **kwargs: Additional arguments that will be passed to the :func:`torch.jit.script` or
 |            :func:`torch.jit.trace` function.
 |
 |      Note:
 |          - Requires the implementation of the
 |            :meth:`~lightning.pytorch.core.LightningModule.forward` method.
 |          - The exported script will be set to evaluation mode.
 |          - It is recommended that you install the latest supported version of PyTorch
 |            to use this feature without limitations. See also the :mod:`torch.jit`
 |            documentation for supported features.
 |
 |      Example::
 |
 |          class SimpleModel(LightningModule):
 |              def __init__(self):
 |                  super().__init__()
 |                  self.l1 = torch.nn.Linear(in_features=64, out_features=4)
 |
 |              def forward(self, x):
 |                  return torch.relu(self.l1(x.view(x.size(0), -1)))
 |
 |          model = SimpleModel()
 |          model.to_torchscript(file_path="model.pt")
 |
 |          torch.jit.save(model.to_torchscript(
 |              file_path="model_trace.pt", method='trace', example_inputs=torch.randn(1, 64))
 |          )
 |
 |      Return:
 |          This LightningModule as a torchscript, regardless of whether `file_path` is
 |          defined or not.
 |
 |  toggle_optimizer(self, optimizer: Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer]) -> None
 |      Makes sure only the gradients of the current optimizer's parameters are calculated in the training step to
 |      prevent dangling gradients in multiple-optimizer setup.
 |
 |      It works with :meth:`untoggle_optimizer` to make sure ``param_requires_grad_state`` is properly reset.
 |
 |      Args:
 |          optimizer: The optimizer to toggle.
 |
 |  unfreeze(self) -> None
 |      Unfreeze all parameters for training.
 |
 |      .. code-block:: python
 |
 |          model = MyLightningModule(...)
 |          model.unfreeze()
 |
 |  untoggle_optimizer(self, optimizer: Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer]) -> None
 |      Resets the state of required gradients that were toggled with :meth:`toggle_optimizer`.
 |
 |      Args:
 |          optimizer: The optimizer to untoggle.
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from lightning.pytorch.core.module.LightningModule:
 |
 |  current_epoch
 |      The current epoch in the ``Trainer``, or 0 if not attached.
 |
 |  device_mesh
 |      Strategies like ``ModelParallelStrategy`` will create a device mesh that can be accessed in the
 |      :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` hook to parallelize the LightningModule.
 |
 |  global_rank
 |      The index of the current process across all nodes and devices.
 |
 |  global_step
 |      Total training batches seen across all epochs.
 |
 |      If no Trainer is attached, this propery is 0.
 |
 |  local_rank
 |      The index of the current process within a single node.
 |
 |  logger
 |      Reference to the logger object in the Trainer.
 |
 |  loggers
 |      Reference to the list of loggers in the Trainer.
 |
 |  on_gpu
 |      Returns ``True`` if this model is currently located on a GPU.
 |
 |      Useful to set flags around the LightningModule for different CPU vs GPU behavior.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from lightning.pytorch.core.module.LightningModule:
 |
 |  automatic_optimization
 |      If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``.
 |
 |  example_input_array
 |      The example input array is a specification of what the module can consume in the :meth:`forward` method. The
 |      return type is interpreted as follows:
 |
 |      -   Single tensor: It is assumed the model takes a single argument, i.e.,
 |          ``model.forward(model.example_input_array)``
 |      -   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,
 |          ``model.forward(*model.example_input_array)``
 |      -   Dict: The input array represents named keyword arguments, i.e.,
 |          ``model.forward(**model.example_input_array)``
 |
 |  fabric
 |
 |  strict_loading
 |      Determines how Lightning loads this model using `.load_state_dict(..., strict=model.strict_loading)`.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from lightning.pytorch.core.module.LightningModule:
 |
 |  CHECKPOINT_HYPER_PARAMS_KEY = 'hyper_parameters'
 |
 |  CHECKPOINT_HYPER_PARAMS_NAME = 'hparams_name'
 |
 |  CHECKPOINT_HYPER_PARAMS_TYPE = 'hparams_type'
 |
 |  __jit_unused_properties__ = ['example_input_array', 'on_gpu', 'current...
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin:
 |
 |  cpu(self) -> Self
 |      See :meth:`torch.nn.Module.cpu`.
 |
 |  cuda(self, device: Union[torch.device, int, NoneType] = None) -> Self
 |      Moves all model parameters and buffers to the GPU. This also makes associated parameters and buffers
 |      different objects. So it should be called before constructing optimizer if the module will live on GPU while
 |      being optimized.
 |
 |      Arguments:
 |          device: If specified, all parameters will be copied to that device. If `None`, the current CUDA device
 |              index will be used.
 |
 |      Returns:
 |          Module: self
 |
 |  double(self) -> Self
 |      See :meth:`torch.nn.Module.double`.
 |
 |  float(self) -> Self
 |      See :meth:`torch.nn.Module.float`.
 |
 |  half(self) -> Self
 |      See :meth:`torch.nn.Module.half`.
 |
 |  to(self, *args: Any, **kwargs: Any) -> Self
 |      See :meth:`torch.nn.Module.to`.
 |
 |  type(self, dst_type: Union[str, torch.dtype]) -> Self
 |      See :meth:`torch.nn.Module.type`.
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin:
 |
 |  device
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin:
 |
 |  dtype
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin:
 |
 |  save_hyperparameters(self, *args: Any, ignore: Union[Sequence[str], str, NoneType] = None, frame: Optional[frame] = None, logger: bool = True) -> None
 |      Save arguments to ``hparams`` attribute.
 |
 |      Args:
 |          args: single object of `dict`, `NameSpace` or `OmegaConf`
 |              or string names or arguments from class ``__init__``
 |          ignore: an argument name or a list of argument names from
 |              class ``__init__`` to be ignored
 |          frame: a frame object. Default is None
 |          logger: Whether to send the hyperparameters to the logger. Default: True
 |
 |      Example::
 |          >>> from lightning.pytorch.core.mixins import HyperparametersMixin
 |          >>> class ManuallyArgsModel(HyperparametersMixin):
 |          ...     def __init__(self, arg1, arg2, arg3):
 |          ...         super().__init__()
 |          ...         # manually assign arguments
 |          ...         self.save_hyperparameters('arg1', 'arg3')
 |          ...     def forward(self, *args, **kwargs):
 |          ...         ...
 |          >>> model = ManuallyArgsModel(1, 'abc', 3.14)
 |          >>> model.hparams
 |          "arg1": 1
 |          "arg3": 3.14
 |
 |          >>> from lightning.pytorch.core.mixins import HyperparametersMixin
 |          >>> class AutomaticArgsModel(HyperparametersMixin):
 |          ...     def __init__(self, arg1, arg2, arg3):
 |          ...         super().__init__()
 |          ...         # equivalent automatic
 |          ...         self.save_hyperparameters()
 |          ...     def forward(self, *args, **kwargs):
 |          ...         ...
 |          >>> model = AutomaticArgsModel(1, 'abc', 3.14)
 |          >>> model.hparams
 |          "arg1": 1
 |          "arg2": abc
 |          "arg3": 3.14
 |
 |          >>> from lightning.pytorch.core.mixins import HyperparametersMixin
 |          >>> class SingleArgModel(HyperparametersMixin):
 |          ...     def __init__(self, params):
 |          ...         super().__init__()
 |          ...         # manually assign single argument
 |          ...         self.save_hyperparameters(params)
 |          ...     def forward(self, *args, **kwargs):
 |          ...         ...
 |          >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
 |          >>> model.hparams
 |          "p1": 1
 |          "p2": abc
 |          "p3": 3.14
 |
 |          >>> from lightning.pytorch.core.mixins import HyperparametersMixin
 |          >>> class ManuallyArgsModel(HyperparametersMixin):
 |          ...     def __init__(self, arg1, arg2, arg3):
 |          ...         super().__init__()
 |          ...         # pass argument(s) to ignore as a string or in a list
 |          ...         self.save_hyperparameters(ignore='arg2')
 |          ...     def forward(self, *args, **kwargs):
 |          ...         ...
 |          >>> model = ManuallyArgsModel(1, 'abc', 3.14)
 |          >>> model.hparams
 |          "arg1": 1
 |          "arg3": 3.14
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin:
 |
 |  hparams_initial
 |      The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.
 |      Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.
 |
 |      Returns:
 |          AttributeDict: immutable initial hyperparameters
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from lightning.pytorch.core.hooks.ModelHooks:
 |
 |  configure_model(self) -> None
 |      Hook to create modules in a strategy and precision aware context.
 |
 |      This is particularly useful for when using sharded strategies (FSDP and DeepSpeed), where we'd like to shard
 |      the model instantly to save memory and initialization time.
 |      For non-sharded strategies, you can choose to override this hook or to initialize your model under the
 |      :meth:`~lightning.pytorch.trainer.trainer.Trainer.init_module` context manager.
 |
 |      This hook is called during each of fit/val/test/predict stages in the same process, so ensure that
 |      implementation of this hook is **idempotent**, i.e., after the first time the hook is called, subsequent calls
 |      to it should be a no-op.
 |
 |  configure_sharded_model(self) -> None
 |      Deprecated.
 |
 |      Use :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` instead.
 |
 |  on_before_backward(self, loss: torch.Tensor) -> None
 |      Called before ``loss.backward()``.
 |
 |      Args:
 |          loss: Loss divided by number of batches for gradient accumulation and scaled if using AMP.
 |
 |  on_before_optimizer_step(self, optimizer: torch.optim.optimizer.Optimizer) -> None
 |      Called before ``optimizer.step()``.
 |
 |      If using gradient accumulation, the hook is called once the gradients have been accumulated.
 |      See: :paramref:`~lightning.pytorch.trainer.trainer.Trainer.accumulate_grad_batches`.
 |
 |      If using AMP, the loss will be unscaled before calling this hook.
 |      See these `docs <https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients>`__
 |      for more information on the scaling of gradients.
 |
 |      If clipping gradients, the gradients will not have been clipped yet.
 |
 |      Args:
 |          optimizer: Current optimizer being used.
 |
 |      Example::
 |
 |          def on_before_optimizer_step(self, optimizer):
 |              # example to inspect gradient information in tensorboard
 |              if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
 |                  for k, v in self.named_parameters():
 |                      self.logger.experiment.add_histogram(
 |                          tag=k, values=v.grad, global_step=self.trainer.global_step
 |                      )
 |
 |  on_before_zero_grad(self, optimizer: torch.optim.optimizer.Optimizer) -> None
 |      Called after ``training_step()`` and before ``optimizer.zero_grad()``.
 |
 |      Called in the training loop after taking an optimizer step and before zeroing grads.
 |      Good place to inspect weight information with weights updated.
 |
 |      This is where it is called::
 |
 |          for optimizer in optimizers:
 |              out = training_step(...)
 |
 |              model.on_before_zero_grad(optimizer) # < ---- called here
 |              optimizer.zero_grad()
 |
 |              backward()
 |
 |      Args:
 |          optimizer: The optimizer for which grads should be zeroed.
 |
 |  on_fit_end(self) -> None
 |      Called at the very end of fit.
 |
 |      If on DDP it is called on every process
 |
 |  on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None
 |      Called in the predict loop after the batch.
 |
 |      Args:
 |          outputs: The outputs of predict_step(x)
 |          batch: The batched data as it is returned by the prediction DataLoader.
 |          batch_idx: the index of the batch
 |          dataloader_idx: the index of the dataloader
 |
 |  on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None
 |      Called in the predict loop before anything happens for that batch.
 |
 |      Args:
 |          batch: The batched data as it is returned by the test DataLoader.
 |          batch_idx: the index of the batch
 |          dataloader_idx: the index of the dataloader
 |
 |  on_predict_epoch_end(self) -> None
 |      Called at the end of predicting.
 |
 |  on_predict_model_eval(self) -> None
 |      Called when the predict loop starts.
 |
 |      The predict loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
 |      to change the behavior.
 |
 |  on_predict_start(self) -> None
 |      Called at the beginning of predicting.
 |
 |  on_test_batch_end(self, outputs: Union[torch.Tensor, Mapping[str, Any], NoneType], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None
 |      Called in the test loop after the batch.
 |
 |      Args:
 |          outputs: The outputs of test_step(x)
 |          batch: The batched data as it is returned by the test DataLoader.
 |          batch_idx: the index of the batch
 |          dataloader_idx: the index of the dataloader
 |
 |  on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None
 |      Called in the test loop before anything happens for that batch.
 |
 |      Args:
 |          batch: The batched data as it is returned by the test DataLoader.
 |          batch_idx: the index of the batch
 |          dataloader_idx: the index of the dataloader
 |
 |  on_test_model_eval(self) -> None
 |      Called when the test loop starts.
 |
 |      The test loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
 |      to change the behavior. See also :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_test_model_train`.
 |
 |  on_test_model_train(self) -> None
 |      Called when the test loop ends.
 |
 |      The test loop by default restores the `training` mode of the LightningModule to what it was before
 |      starting testing. Override this hook to change the behavior. See also
 |      :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_test_model_eval`.
 |
 |  on_test_start(self) -> None
 |      Called at the beginning of testing.
 |
 |  on_validation_batch_end(self, outputs: Union[torch.Tensor, Mapping[str, Any], NoneType], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None
 |      Called in the validation loop after the batch.
 |
 |      Args:
 |          outputs: The outputs of validation_step(x)
 |          batch: The batched data as it is returned by the validation DataLoader.
 |          batch_idx: the index of the batch
 |          dataloader_idx: the index of the dataloader
 |
 |  on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None
 |      Called in the validation loop before anything happens for that batch.
 |
 |      Args:
 |          batch: The batched data as it is returned by the validation DataLoader.
 |          batch_idx: the index of the batch
 |          dataloader_idx: the index of the dataloader
 |
 |  on_validation_end(self) -> None
 |      Called at the end of validation.
 |
 |  on_validation_model_eval(self) -> None
 |      Called when the validation loop starts.
 |
 |      The validation loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
 |      to change the behavior. See also :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_validation_model_train`.
 |
 |  on_validation_model_train(self) -> None
 |      Called when the validation loop ends.
 |
 |      The validation loop by default restores the `training` mode of the LightningModule to what it was before
 |      starting validation. Override this hook to change the behavior. See also
 |      :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_validation_model_eval`.
 |
 |  on_validation_model_zero_grad(self) -> None
 |      Called by the training loop to release gradients before entering the validation loop.
 |
 |  on_validation_start(self) -> None
 |      Called at the beginning of validation.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from lightning.pytorch.core.hooks.DataHooks:
 |
 |  on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any
 |      Override to alter or apply batch augmentations to your batch after it is transferred to the device.
 |
 |      Note:
 |          To check the current state of execution of this hook you can use
 |          ``self.trainer.training/testing/validating/predicting`` so that you can
 |          add different logic as per your requirement.
 |
 |      Args:
 |          batch: A batch of data that needs to be altered or augmented.
 |          dataloader_idx: The index of the dataloader to which the batch belongs.
 |
 |      Returns:
 |          A batch of data
 |
 |      Example::
 |
 |          def on_after_batch_transfer(self, batch, dataloader_idx):
 |              batch['x'] = gpu_transforms(batch['x'])
 |              return batch
 |
 |      See Also:
 |          - :meth:`on_before_batch_transfer`
 |          - :meth:`transfer_batch_to_device`
 |
 |  on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any
 |      Override to alter or apply batch augmentations to your batch before it is transferred to the device.
 |
 |      Note:
 |          To check the current state of execution of this hook you can use
 |          ``self.trainer.training/testing/validating/predicting`` so that you can
 |          add different logic as per your requirement.
 |
 |      Args:
 |          batch: A batch of data that needs to be altered or augmented.
 |          dataloader_idx: The index of the dataloader to which the batch belongs.
 |
 |      Returns:
 |          A batch of data
 |
 |      Example::
 |
 |          def on_before_batch_transfer(self, batch, dataloader_idx):
 |              batch['x'] = transforms(batch['x'])
 |              return batch
 |
 |      See Also:
 |          - :meth:`on_after_batch_transfer`
 |          - :meth:`transfer_batch_to_device`
 |
 |  predict_dataloader(self) -> Any
 |      An iterable or collection of iterables specifying prediction samples.
 |
 |      For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.
 |
 |      It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.
 |
 |      - :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`
 |      - :meth:`prepare_data`
 |      - :meth:`setup`
 |
 |      Note:
 |          Lightning tries to add the correct sampler for distributed and arbitrary hardware
 |          There is no need to set it yourself.
 |
 |      Return:
 |          A :class:`torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.
 |
 |  prepare_data(self) -> None
 |      Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
 |      settings) will result in corrupted data. Lightning ensures this method is called only within a single process,
 |      so you can safely add your downloading logic within.
 |
 |      .. warning:: DO NOT set state to the model (use ``setup`` instead)
 |          since this is NOT called on every device
 |
 |      Example::
 |
 |          def prepare_data(self):
 |              # good
 |              download_data()
 |              tokenize()
 |              etc()
 |
 |              # bad
 |              self.split = data_split
 |              self.some_state = some_other_state()
 |
 |      In a distributed environment, ``prepare_data`` can be called in two ways
 |      (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)
 |
 |      1. Once per node. This is the default and is only called on LOCAL_RANK=0.
 |      2. Once in total. Only called on GLOBAL_RANK=0.
 |
 |      Example::
 |
 |          # DEFAULT
 |          # called once per node on LOCAL_RANK=0 of that node
 |          class LitDataModule(LightningDataModule):
 |              def __init__(self):
 |                  super().__init__()
 |                  self.prepare_data_per_node = True
 |
 |
 |          # call on GLOBAL_RANK=0 (great for shared file systems)
 |          class LitDataModule(LightningDataModule):
 |              def __init__(self):
 |                  super().__init__()
 |                  self.prepare_data_per_node = False
 |
 |      This is called before requesting the dataloaders:
 |
 |      .. code-block:: python
 |
 |          model.prepare_data()
 |          initialize_distributed()
 |          model.setup(stage)
 |          model.train_dataloader()
 |          model.val_dataloader()
 |          model.test_dataloader()
 |          model.predict_dataloader()
 |
 |  transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any
 |      Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors wrapped in a custom data
 |      structure.
 |
 |      The data types listed below (and any arbitrary nesting of them) are supported out of the box:
 |
 |      - :class:`torch.Tensor` or anything that implements `.to(...)`
 |      - :class:`list`
 |      - :class:`dict`
 |      - :class:`tuple`
 |
 |      For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).
 |
 |      Note:
 |          This hook should only transfer the data and not modify it, nor should it move the data to
 |          any other device than the one passed in as argument (unless you know what you are doing).
 |          To check the current state of execution of this hook you can use
 |          ``self.trainer.training/testing/validating/predicting`` so that you can
 |          add different logic as per your requirement.
 |
 |      Args:
 |          batch: A batch of data that needs to be transferred to a new device.
 |          device: The target device as defined in PyTorch.
 |          dataloader_idx: The index of the dataloader to which the batch belongs.
 |
 |      Returns:
 |          A reference to the data on the new device.
 |
 |      Example::
 |
 |          def transfer_batch_to_device(self, batch, device, dataloader_idx):
 |              if isinstance(batch, CustomBatch):
 |                  # move all tensors in your custom data structure to the device
 |                  batch.samples = batch.samples.to(device)
 |                  batch.targets = batch.targets.to(device)
 |              elif dataloader_idx == 0:
 |                  # skip device transfer for the first dataloader or anything you wish
 |                  pass
 |              else:
 |                  batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
 |              return batch
 |
 |      See Also:
 |          - :meth:`move_data_to_device`
 |          - :meth:`apply_to_collection`
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from lightning.pytorch.core.hooks.CheckpointHooks:
 |
 |  on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None
 |      Called by Lightning to restore your model. If you saved something with :meth:`on_save_checkpoint` this is
 |      your chance to restore this.
 |
 |      Args:
 |          checkpoint: Loaded checkpoint
 |
 |      Example::
 |
 |          def on_load_checkpoint(self, checkpoint):
 |              # 99% of the time you don't need to implement this method
 |              self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']
 |
 |      Note:
 |          Lightning auto-restores global step, epoch, and train state including amp scaling.
 |          There is no need for you to restore anything regarding training.
 |
 |  on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None
 |      Called by Lightning when saving a checkpoint to give you a chance to store anything else you might want to
 |      save.
 |
 |      Args:
 |          checkpoint: The full checkpoint dictionary before it gets dumped to a file.
 |              Implementations of this hook can insert additional data into this dictionary.
 |
 |      Example::
 |
 |          def on_save_checkpoint(self, checkpoint):
 |              # 99% of use cases you don't need to implement this method
 |              checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object
 |
 |      Note:
 |          Lightning saves all aspects of training (epoch, global step, etc...)
 |          including amp scaling.
 |          There is no need for you to store anything about training.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from torch.nn.modules.module.Module:
 |
 |  __call__ = _wrapped_call_impl(self, *args, **kwargs)
 |
 |  __delattr__(self, name) -> None
 |      Implement delattr(self, name).
 |
 |  __dir__(self)
 |      Default dir() implementation.
 |
 |  __getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]
 |      # It is crucial that the return type is not annotated as `Any`, otherwise type checking
 |      # on `torch.nn.Module` and all its subclasses is largely disabled as a result. See:
 |      # https://github.com/pytorch/pytorch/pull/115074
 |
 |  __repr__(self) -> str
 |      Return repr(self).
 |
 |  __setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None
 |      Implement setattr(self, name, value).
 |
 |  __setstate__(self, state)
 |
 |  add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None
 |      Add a child module to the current module.
 |
 |      The module can be accessed as an attribute using the given name.
 |
 |      Args:
 |          name (str): name of the child module. The child module can be
 |              accessed from this module using the given name
 |          module (Module): child module to be added to the module.
 |
 |  apply(self, fn: collections.abc.Callable[['Module'], None]) -> Self
 |      Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.
 |
 |      Typical use includes initializing the parameters of a model
 |      (see also :ref:`nn-init-doc`).
 |
 |      Args:
 |          fn (:class:`Module` -> None): function to be applied to each submodule
 |
 |      Returns:
 |          Module: self
 |
 |      Example::
 |
 |          >>> @torch.no_grad()
 |          >>> def init_weights(m):
 |          >>>     print(m)
 |          >>>     if type(m) is nn.Linear:
 |          >>>         m.weight.fill_(1.0)
 |          >>>         print(m.weight)
 |          >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
 |          >>> net.apply(init_weights)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          Parameter containing:
 |          tensor([[1., 1.],
 |                  [1., 1.]], requires_grad=True)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          Parameter containing:
 |          tensor([[1., 1.],
 |                  [1., 1.]], requires_grad=True)
 |          Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          )
 |
 |  bfloat16(self) -> Self
 |      Casts all floating point parameters and buffers to ``bfloat16`` datatype.
 |
 |      .. note::
 |          This method modifies the module in-place.
 |
 |      Returns:
 |          Module: self
 |
 |  buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
 |      Return an iterator over module buffers.
 |
 |      Args:
 |          recurse (bool): if True, then yields buffers of this module
 |              and all submodules. Otherwise, yields only buffers that
 |              are direct members of this module.
 |
 |      Yields:
 |          torch.Tensor: module buffer
 |
 |      Example::
 |
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for buf in model.buffers():
 |          >>>     print(type(buf), buf.size())
 |          <class 'torch.Tensor'> (20L,)
 |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
 |
 |  children(self) -> collections.abc.Iterator['Module']
 |      Return an iterator over immediate children modules.
 |
 |      Yields:
 |          Module: a child module
 |
 |  compile(self, *args, **kwargs) -> None
 |      Compile this Module's forward using :func:`torch.compile`.
 |
 |      This Module's `__call__` method is compiled and all arguments are passed as-is
 |      to :func:`torch.compile`.
 |
 |      See :func:`torch.compile` for details on the arguments for this function.
 |
 |  eval(self) -> Self
 |      Set the module in evaluation mode.
 |
 |      This has an effect only on certain modules. See the documentation of
 |      particular modules for details of their behaviors in training/evaluation
 |      mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
 |      etc.
 |
 |      This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.
 |
 |      See :ref:`locally-disable-grad-doc` for a comparison between
 |      `.eval()` and several similar mechanisms that may be confused with it.
 |
 |      Returns:
 |          Module: self
 |
 |  extra_repr(self) -> str
 |      Return the extra representation of the module.
 |
 |      To print customized extra information, you should re-implement
 |      this method in your own modules. Both single-line and multi-line
 |      strings are acceptable.
 |
 |  get_buffer(self, target: str) -> 'Tensor'
 |      Return the buffer given by ``target`` if it exists, otherwise throw an error.
 |
 |      See the docstring for ``get_submodule`` for a more detailed
 |      explanation of this method's functionality as well as how to
 |      correctly specify ``target``.
 |
 |      Args:
 |          target: The fully-qualified string name of the buffer
 |              to look for. (See ``get_submodule`` for how to specify a
 |              fully-qualified string.)
 |
 |      Returns:
 |          torch.Tensor: The buffer referenced by ``target``
 |
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not a
 |              buffer
 |
 |  get_extra_state(self) -> Any
 |      Return any extra state to include in the module's state_dict.
 |
 |      Implement this and a corresponding :func:`set_extra_state` for your module
 |      if you need to store extra state. This function is called when building the
 |      module's `state_dict()`.
 |
 |      Note that extra state should be picklable to ensure working serialization
 |      of the state_dict. We only provide backwards compatibility guarantees
 |      for serializing Tensors; other objects may break backwards compatibility if
 |      their serialized pickled form changes.
 |
 |      Returns:
 |          object: Any extra state to store in the module's state_dict
 |
 |  get_parameter(self, target: str) -> 'Parameter'
 |      Return the parameter given by ``target`` if it exists, otherwise throw an error.
 |
 |      See the docstring for ``get_submodule`` for a more detailed
 |      explanation of this method's functionality as well as how to
 |      correctly specify ``target``.
 |
 |      Args:
 |          target: The fully-qualified string name of the Parameter
 |              to look for. (See ``get_submodule`` for how to specify a
 |              fully-qualified string.)
 |
 |      Returns:
 |          torch.nn.Parameter: The Parameter referenced by ``target``
 |
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not an
 |              ``nn.Parameter``
 |
 |  get_submodule(self, target: str) -> 'Module'
 |      Return the submodule given by ``target`` if it exists, otherwise throw an error.
 |
 |      For example, let's say you have an ``nn.Module`` ``A`` that
 |      looks like this:
 |
 |      .. code-block:: text
 |
 |          A(
 |              (net_b): Module(
 |                  (net_c): Module(
 |                      (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
 |                  )
 |                  (linear): Linear(in_features=100, out_features=200, bias=True)
 |              )
 |          )
 |
 |      (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
 |      submodule ``net_b``, which itself has two submodules ``net_c``
 |      and ``linear``. ``net_c`` then has a submodule ``conv``.)
 |
 |      To check whether or not we have the ``linear`` submodule, we
 |      would call ``get_submodule("net_b.linear")``. To check whether
 |      we have the ``conv`` submodule, we would call
 |      ``get_submodule("net_b.net_c.conv")``.
 |
 |      The runtime of ``get_submodule`` is bounded by the degree
 |      of module nesting in ``target``. A query against
 |      ``named_modules`` achieves the same result, but it is O(N) in
 |      the number of transitive modules. So, for a simple check to see
 |      if some submodule exists, ``get_submodule`` should always be
 |      used.
 |
 |      Args:
 |          target: The fully-qualified string name of the submodule
 |              to look for. (See above example for how to specify a
 |              fully-qualified string.)
 |
 |      Returns:
 |          torch.nn.Module: The submodule referenced by ``target``
 |
 |      Raises:
 |          AttributeError: If at any point along the path resulting from
 |              the target string the (sub)path resolves to a non-existent
 |              attribute name or an object that is not an instance of ``nn.Module``.
 |
 |  ipu(self, device: int | torch.device | None = None) -> Self
 |      Move all model parameters and buffers to the IPU.
 |
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing the optimizer if the module will
 |      live on IPU while being optimized.
 |
 |      .. note::
 |          This method modifies the module in-place.
 |
 |      Arguments:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |
 |      Returns:
 |          Module: self
 |
 |  load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
 |      Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.
 |
 |      If :attr:`strict` is ``True``, then
 |      the keys of :attr:`state_dict` must exactly match the keys returned
 |      by this module's :meth:`~torch.nn.Module.state_dict` function.
 |
 |      .. warning::
 |          If :attr:`assign` is ``True`` the optimizer must be created after
 |          the call to :attr:`load_state_dict` unless
 |          :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.
 |
 |      Args:
 |          state_dict (dict): a dict containing parameters and
 |              persistent buffers.
 |          strict (bool, optional): whether to strictly enforce that the keys
 |              in :attr:`state_dict` match the keys returned by this module's
 |              :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
 |          assign (bool, optional): When set to ``False``, the properties of the tensors
 |              in the current module are preserved whereas setting it to ``True`` preserves
 |              properties of the Tensors in the state dict. The only
 |              exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`
 |              for which the value from the module is preserved. Default: ``False``
 |
 |      Returns:
 |          ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
 |              * ``missing_keys`` is a list of str containing any keys that are expected
 |                  by this module but missing from the provided ``state_dict``.
 |              * ``unexpected_keys`` is a list of str containing the keys that are not
 |                  expected by this module but present in the provided ``state_dict``.
 |
 |      Note:
 |          If a parameter or buffer is registered as ``None`` and its corresponding key
 |          exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
 |          ``RuntimeError``.
 |
 |  modules(self) -> collections.abc.Iterator['Module']
 |      Return an iterator over all modules in the network.
 |
 |      Yields:
 |          Module: a module in the network
 |
 |      Note:
 |          Duplicate modules are returned only once. In the following
 |          example, ``l`` will be returned only once.
 |
 |      Example::
 |
 |          >>> l = nn.Linear(2, 2)
 |          >>> net = nn.Sequential(l, l)
 |          >>> for idx, m in enumerate(net.modules()):
 |          ...     print(idx, '->', m)
 |
 |          0 -> Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          )
 |          1 -> Linear(in_features=2, out_features=2, bias=True)
 |
 |  mtia(self, device: int | torch.device | None = None) -> Self
 |      Move all model parameters and buffers to the MTIA.
 |
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing the optimizer if the module will
 |      live on MTIA while being optimized.
 |
 |      .. note::
 |          This method modifies the module in-place.
 |
 |      Arguments:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |
 |      Returns:
 |          Module: self
 |
 |  named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
 |      Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.
 |
 |      Args:
 |          prefix (str): prefix to prepend to all buffer names.
 |          recurse (bool, optional): if True, then yields buffers of this module
 |              and all submodules. Otherwise, yields only buffers that
 |              are direct members of this module. Defaults to True.
 |          remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.
 |
 |      Yields:
 |          (str, torch.Tensor): Tuple containing the name and buffer
 |
 |      Example::
 |
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for name, buf in self.named_buffers():
 |          >>>     if name in ['running_var']:
 |          >>>         print(buf.size())
 |
 |  named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]
 |      Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
 |
 |      Yields:
 |          (str, Module): Tuple containing a name and child module
 |
 |      Example::
 |
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for name, module in model.named_children():
 |          >>>     if name in ['conv4', 'conv5']:
 |          >>>         print(module)
 |
 |  named_modules(self, memo: set['Module'] | None = None, prefix: str = '', remove_duplicate: bool = True)
 |      Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.
 |
 |      Args:
 |          memo: a memo to store the set of modules already added to the result
 |          prefix: a prefix that will be added to the name of the module
 |          remove_duplicate: whether to remove the duplicated module instances in the result
 |              or not
 |
 |      Yields:
 |          (str, Module): Tuple of name and module
 |
 |      Note:
 |          Duplicate modules are returned only once. In the following
 |          example, ``l`` will be returned only once.
 |
 |      Example::
 |
 |          >>> l = nn.Linear(2, 2)
 |          >>> net = nn.Sequential(l, l)
 |          >>> for idx, m in enumerate(net.named_modules()):
 |          ...     print(idx, '->', m)
 |
 |          0 -> ('', Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          ))
 |          1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
 |
 |  named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
 |      Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
 |
 |      Args:
 |          prefix (str): prefix to prepend to all parameter names.
 |          recurse (bool): if True, then yields parameters of this module
 |              and all submodules. Otherwise, yields only parameters that
 |              are direct members of this module.
 |          remove_duplicate (bool, optional): whether to remove the duplicated
 |              parameters in the result. Defaults to True.
 |
 |      Yields:
 |          (str, Parameter): Tuple containing the name and parameter
 |
 |      Example::
 |
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for name, param in self.named_parameters():
 |          >>>     if name in ['bias']:
 |          >>>         print(param.size())
 |
 |  parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
 |      Return an iterator over module parameters.
 |
 |      This is typically passed to an optimizer.
 |
 |      Args:
 |          recurse (bool): if True, then yields parameters of this module
 |              and all submodules. Otherwise, yields only parameters that
 |              are direct members of this module.
 |
 |      Yields:
 |          Parameter: module parameter
 |
 |      Example::
 |
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for param in model.parameters():
 |          >>>     print(type(param), param.size())
 |          <class 'torch.Tensor'> (20L,)
 |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
 |
 |  register_backward_hook(self, hook: collections.abc.Callable[['Module', typing.Union[tuple[torch.Tensor, ...], torch.Tensor], typing.Union[tuple[torch.Tensor, ...], torch.Tensor]], typing.Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle
 |      Register a backward hook on the module.
 |
 |      This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
 |      the behavior of this function will change in future versions.
 |
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |
 |  register_buffer(self, name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
 |      Add a buffer to the module.
 |
 |      This is typically used to register a buffer that should not be
 |      considered a model parameter. For example, BatchNorm's ``running_mean``
 |      is not a parameter, but is part of the module's state. Buffers, by
 |      default, are persistent and will be saved alongside parameters. This
 |      behavior can be changed by setting :attr:`persistent` to ``False``. The
 |      only difference between a persistent buffer and a non-persistent buffer
 |      is that the latter will not be a part of this module's
 |      :attr:`state_dict`.
 |
 |      Buffers can be accessed as attributes using given names.
 |
 |      Args:
 |          name (str): name of the buffer. The buffer can be accessed
 |              from this module using the given name
 |          tensor (Tensor or None): buffer to be registered. If ``None``, then operations
 |              that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
 |              the buffer is **not** included in the module's :attr:`state_dict`.
 |          persistent (bool): whether the buffer is part of this module's
 |              :attr:`state_dict`.
 |
 |      Example::
 |
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> self.register_buffer('running_mean', torch.zeros(num_features))
 |
 |  register_forward_hook(self, hook: collections.abc.Callable[[~T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[~T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
 |      Register a forward hook on the module.
 |
 |      The hook will be called every time after :func:`forward` has computed an output.
 |
 |      If ``with_kwargs`` is ``False`` or not specified, the input contains only
 |      the positional arguments given to the module. Keyword arguments won't be
 |      passed to the hooks and only to the ``forward``. The hook can modify the
 |      output. It can modify the input inplace but it will not have effect on
 |      forward since this is called after :func:`forward` is called. The hook
 |      should have the following signature::
 |
 |          hook(module, args, output) -> None or modified output
 |
 |      If ``with_kwargs`` is ``True``, the forward hook will be passed the
 |      ``kwargs`` given to the forward function and be expected to return the
 |      output possibly modified. The hook should have the following signature::
 |
 |          hook(module, args, kwargs, output) -> None or modified output
 |
 |      Args:
 |          hook (Callable): The user defined hook to be registered.
 |          prepend (bool): If ``True``, the provided ``hook`` will be fired
 |              before all existing ``forward`` hooks on this
 |              :class:`torch.nn.Module`. Otherwise, the provided
 |              ``hook`` will be fired after all existing ``forward`` hooks on
 |              this :class:`torch.nn.Module`. Note that global
 |              ``forward`` hooks registered with
 |              :func:`register_module_forward_hook` will fire before all hooks
 |              registered by this method.
 |              Default: ``False``
 |          with_kwargs (bool): If ``True``, the ``hook`` will be passed the
 |              kwargs given to the forward function.
 |              Default: ``False``
 |          always_call (bool): If ``True`` the ``hook`` will be run regardless of
 |              whether an exception is raised while calling the Module.
 |              Default: ``False``
 |
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |
 |  register_forward_pre_hook(self, hook: collections.abc.Callable[[~T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[~T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
 |      Register a forward pre-hook on the module.
 |
 |      The hook will be called every time before :func:`forward` is invoked.
 |
 |
 |      If ``with_kwargs`` is false or not specified, the input contains only
 |      the positional arguments given to the module. Keyword arguments won't be
 |      passed to the hooks and only to the ``forward``. The hook can modify the
 |      input. User can either return a tuple or a single modified value in the
 |      hook. We will wrap the value into a tuple if a single value is returned
 |      (unless that value is already a tuple). The hook should have the
 |      following signature::
 |
 |          hook(module, args) -> None or modified input
 |
 |      If ``with_kwargs`` is true, the forward pre-hook will be passed the
 |      kwargs given to the forward function. And if the hook modifies the
 |      input, both the args and kwargs should be returned. The hook should have
 |      the following signature::
 |
 |          hook(module, args, kwargs) -> None or a tuple of modified input and kwargs
 |
 |      Args:
 |          hook (Callable): The user defined hook to be registered.
 |          prepend (bool): If true, the provided ``hook`` will be fired before
 |              all existing ``forward_pre`` hooks on this
 |              :class:`torch.nn.Module`. Otherwise, the provided
 |              ``hook`` will be fired after all existing ``forward_pre`` hooks
 |              on this :class:`torch.nn.Module`. Note that global
 |              ``forward_pre`` hooks registered with
 |              :func:`register_module_forward_pre_hook` will fire before all
 |              hooks registered by this method.
 |              Default: ``False``
 |          with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
 |              given to the forward function.
 |              Default: ``False``
 |
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |
 |  register_full_backward_hook(self, hook: collections.abc.Callable[['Module', typing.Union[tuple[torch.Tensor, ...], torch.Tensor], typing.Union[tuple[torch.Tensor, ...], torch.Tensor]], typing.Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
 |      Register a backward hook on the module.
 |
 |      The hook will be called every time the gradients with respect to a module are computed, and its firing rules are as follows:
 |
 |          1. Ordinarily, the hook fires when the gradients are computed with respect to the module inputs.
 |          2. If none of the module inputs require gradients, the hook will fire when the gradients are computed
 |             with respect to module outputs.
 |          3. If none of the module outputs require gradients, then the hooks will not fire.
 |
 |      The hook should have the following signature::
 |
 |          hook(module, grad_input, grad_output) -> tuple(Tensor) or None
 |
 |      The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
 |      with respect to the inputs and outputs respectively. The hook should
 |      not modify its arguments, but it can optionally return a new gradient with
 |      respect to the input that will be used in place of :attr:`grad_input` in
 |      subsequent computations. :attr:`grad_input` will only correspond to the inputs given
 |      as positional arguments and all kwarg arguments are ignored. Entries
 |      in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
 |      arguments.
 |
 |      For technical reasons, when this hook is applied to a Module, its forward function will
 |      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
 |      of each Tensor returned by the Module's forward function.
 |
 |      .. warning ::
 |          Modifying inputs or outputs inplace is not allowed when using backward hooks and
 |          will raise an error.
 |
 |      Args:
 |          hook (Callable): The user-defined hook to be registered.
 |          prepend (bool): If true, the provided ``hook`` will be fired before
 |              all existing ``backward`` hooks on this
 |              :class:`torch.nn.Module`. Otherwise, the provided
 |              ``hook`` will be fired after all existing ``backward`` hooks on
 |              this :class:`torch.nn.Module`. Note that global
 |              ``backward`` hooks registered with
 |              :func:`register_module_full_backward_hook` will fire before
 |              all hooks registered by this method.
 |
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |
 |  register_full_backward_pre_hook(self, hook: collections.abc.Callable[['Module', typing.Union[tuple[torch.Tensor, ...], torch.Tensor]], typing.Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
 |      Register a backward pre-hook on the module.
 |
 |      The hook will be called every time the gradients for the module are computed.
 |      The hook should have the following signature::
 |
 |          hook(module, grad_output) -> tuple[Tensor, ...], Tensor or None
 |
 |      The :attr:`grad_output` is a tuple. The hook should
 |      not modify its arguments, but it can optionally return a new gradient with
 |      respect to the output that will be used in place of :attr:`grad_output` in
 |      subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
 |      all non-Tensor arguments.
 |
 |      For technical reasons, when this hook is applied to a Module, its forward function will
 |      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
 |      of each Tensor returned by the Module's forward function.
 |
 |      .. warning ::
 |          Modifying inputs inplace is not allowed when using backward hooks and
 |          will raise an error.
 |
 |      Args:
 |          hook (Callable): The user-defined hook to be registered.
 |          prepend (bool): If true, the provided ``hook`` will be fired before
 |              all existing ``backward_pre`` hooks on this
 |              :class:`torch.nn.Module`. Otherwise, the provided
 |              ``hook`` will be fired after all existing ``backward_pre`` hooks
 |              on this :class:`torch.nn.Module`. Note that global
 |              ``backward_pre`` hooks registered with
 |              :func:`register_module_full_backward_pre_hook` will fire before
 |              all hooks registered by this method.
 |
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |
 |  register_load_state_dict_post_hook(self, hook)
 |      Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.
 |
 |      It should have the following signature::
 |          hook(module, incompatible_keys) -> None
 |
 |      The ``module`` argument is the current module that this hook is registered
 |      on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
 |      of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
 |      is a ``list`` of ``str`` containing the missing keys and
 |      ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.
 |
 |      The given incompatible_keys can be modified inplace if needed.
 |
 |      Note that the checks performed when calling :func:`load_state_dict` with
 |      ``strict=True`` are affected by modifications the hook makes to
 |      ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
 |      set of keys will result in an error being thrown when ``strict=True``, and
 |      clearing out both missing and unexpected keys will avoid an error.
 |
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |
 |  register_load_state_dict_pre_hook(self, hook)
 |      Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.
 |
 |      It should have the following signature::
 |          hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950
 |
 |      Arguments:
 |          hook (Callable): Callable hook that will be invoked before
 |              loading the state dict.
 |
 |  register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None
 |      Alias for :func:`add_module`.
 |
 |  register_parameter(self, name: str, param: torch.nn.parameter.Parameter | None) -> None
 |      Add a parameter to the module.
 |
 |      The parameter can be accessed as an attribute using given name.
 |
 |      Args:
 |          name (str): name of the parameter. The parameter can be accessed
 |              from this module using the given name
 |          param (Parameter or None): parameter to be added to the module. If
 |              ``None``, then operations that run on parameters, such as :attr:`cuda`,
 |              are ignored. If ``None``, the parameter is **not** included in the
 |              module's :attr:`state_dict`.
 |
 |  register_state_dict_post_hook(self, hook)
 |      Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.
 |
 |      It should have the following signature::
 |          hook(module, state_dict, prefix, local_metadata) -> None
 |
 |      The registered hooks can modify the ``state_dict`` inplace.
 |
 |  register_state_dict_pre_hook(self, hook)
 |      Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.
 |
 |      It should have the following signature::
 |          hook(module, prefix, keep_vars) -> None
 |
 |      The registered hooks can be used to perform pre-processing before the ``state_dict``
 |      call is made.
 |
 |  requires_grad_(self, requires_grad: bool = True) -> Self
 |      Change if autograd should record operations on parameters in this module.
 |
 |      This method sets the parameters' :attr:`requires_grad` attributes
 |      in-place.
 |
 |      This method is helpful for freezing part of the module for finetuning
 |      or training parts of a model individually (e.g., GAN training).
 |
 |      See :ref:`locally-disable-grad-doc` for a comparison between
 |      `.requires_grad_()` and several similar mechanisms that may be confused with it.
 |
 |      Args:
 |          requires_grad (bool): whether autograd should record operations on
 |                                parameters in this module. Default: ``True``.
 |
 |      Returns:
 |          Module: self
 |
 |  set_extra_state(self, state: Any) -> None
 |      Set extra state contained in the loaded `state_dict`.
 |
 |      This function is called from :func:`load_state_dict` to handle any extra state
 |      found within the `state_dict`. Implement this function and a corresponding
 |      :func:`get_extra_state` for your module if you need to store extra state within its
 |      `state_dict`.
 |
 |      Args:
 |          state (dict): Extra state from the `state_dict`
 |
 |  set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None
 |      Set the submodule given by ``target`` if it exists, otherwise throw an error.
 |
 |      .. note::
 |          If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
 |          or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
 |          the method will only attempt to replace an existing submodule and throw an error if
 |          the submodule does not exist.
 |
 |      For example, let's say you have an ``nn.Module`` ``A`` that
 |      looks like this:
 |
 |      .. code-block:: text
 |
 |          A(
 |              (net_b): Module(
 |                  (net_c): Module(
 |                      (conv): Conv2d(3, 3, 3)
 |                  )
 |                  (linear): Linear(3, 3)
 |              )
 |          )
 |
 |      (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
 |      submodule ``net_b``, which itself has two submodules ``net_c``
 |      and ``linear``. ``net_c`` then has a submodule ``conv``.)
 |
 |      To override the ``Conv2d`` with a new submodule ``Linear``, you
 |      could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
 |      where ``strict`` could be ``True`` or ``False``
 |
 |      To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
 |      you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.
 |
 |      In the above if you set ``strict=True`` and call
 |      ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
 |      will be raised because ``net_b`` does not have a submodule named ``conv``.
 |
 |      Args:
 |          target: The fully-qualified string name of the submodule
 |              to look for. (See above example for how to specify a
 |              fully-qualified string.)
 |          module: The module to set the submodule to.
 |          strict: If ``False``, the method will replace an existing submodule
 |              or create a new submodule if the parent module exists. If ``True``,
 |              the method will only attempt to replace an existing submodule and throw an error
 |              if the submodule doesn't already exist.
 |
 |      Raises:
 |          ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
 |          AttributeError: If at any point along the path resulting from
 |              the ``target`` string the (sub)path resolves to a non-existent
 |              attribute name or an object that is not an instance of ``nn.Module``.
 |
 |  share_memory(self) -> Self
 |      See :meth:`torch.Tensor.share_memory_`.
 |
 |  state_dict(self, *args, destination=None, prefix='', keep_vars=False)
 |      Return a dictionary containing references to the whole state of the module.
 |
 |      Both parameters and persistent buffers (e.g. running averages) are
 |      included. Keys are corresponding parameter and buffer names.
 |      Parameters and buffers set to ``None`` are not included.
 |
 |      .. note::
 |          The returned object is a shallow copy. It contains references
 |          to the module's parameters and buffers.
 |
 |      .. warning::
 |          Currently ``state_dict()`` also accepts positional arguments for
 |          ``destination``, ``prefix`` and ``keep_vars`` in order. However,
 |          this is being deprecated and keyword arguments will be enforced in
 |          future releases.
 |
 |      .. warning::
 |          Please avoid the use of argument ``destination`` as it is not
 |          designed for end-users.
 |
 |      Args:
 |          destination (dict, optional): If provided, the state of module will
 |              be updated into the dict and the same object is returned.
 |              Otherwise, an ``OrderedDict`` will be created and returned.
 |              Default: ``None``.
 |          prefix (str, optional): a prefix added to parameter and buffer
 |              names to compose the keys in state_dict. Default: ``''``.
 |          keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
 |              returned in the state dict are detached from autograd. If it's
 |              set to ``True``, detaching will not be performed.
 |              Default: ``False``.
 |
 |      Returns:
 |          dict:
 |              a dictionary containing a whole state of the module
 |
 |      Example::
 |
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> module.state_dict().keys()
 |          ['bias', 'weight']
 |
 |  to_empty(self, *, device: Union[str, torch.device, int, NoneType], recurse: bool = True) -> Self
 |      Move the parameters and buffers to the specified device without copying storage.
 |
 |      Args:
 |          device (:class:`torch.device`): The desired device of the parameters
 |              and buffers in this module.
 |          recurse (bool): Whether parameters and buffers of submodules should
 |              be recursively moved to the specified device.
 |
 |      Returns:
 |          Module: self
 |
 |  train(self, mode: bool = True) -> Self
 |      Set the module in training mode.
 |
 |      This has an effect only on certain modules. See the documentation of
 |      particular modules for details of their behaviors in training/evaluation
 |      mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
 |      etc.
 |
 |      Args:
 |          mode (bool): whether to set training mode (``True``) or evaluation
 |                       mode (``False``). Default: ``True``.
 |
 |      Returns:
 |          Module: self
 |
 |  xpu(self, device: int | torch.device | None = None) -> Self
 |      Move all model parameters and buffers to the XPU.
 |
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing optimizer if the module will
 |      live on XPU while being optimized.
 |
 |      .. note::
 |          This method modifies the module in-place.
 |
 |      Arguments:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |
 |      Returns:
 |          Module: self
 |
 |  zero_grad(self, set_to_none: bool = True) -> None
 |      Reset gradients of all model parameters.
 |
 |      See similar function under :class:`torch.optim.Optimizer` for more context.
 |
 |      Args:
 |          set_to_none (bool): instead of setting to zero, set the grads to None.
 |              See :meth:`torch.optim.Optimizer.zero_grad` for details.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from torch.nn.modules.module.Module:
 |
 |  T_destination = ~T_destination
 |
 |  call_super_init = False
 |
 |  dump_patches = False
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.common.Model:
 |
 |  generate_model_card(self, type: 'str' = 'hf', template: 'str' = None, template_kwargs: 'Optional[Dict[str, str]]' = None) -> 'object'
 |      Generates a ModelCard for the current model. This method is called when pushing the model to the Hub.
 |
 |      Returns:
 |          An object that can be represented as a str representation of the model card, usually in Markdown format.
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from nemo.core.classes.common.Model:
 |
 |  from_pretrained(model_name: 'str', refresh_cache: 'bool' = False, override_config_path: 'Optional[str]' = None, map_location: "Optional['torch.device']" = None, strict: 'bool' = True, return_config: 'bool' = False, trainer: "Optional['Trainer']" = None, save_restore_connector: 'SaveRestoreConnector' = None, return_model_file: 'Optional[bool]' = False)
 |      Instantiates an instance of NeMo from NVIDIA NGC cloud
 |      Use restore_from() to instantiate from a local .nemo file.
 |      Args:
 |          model_name: string key which will be used to find the module.
 |          refresh_cache: If set to True, then when fetching from cloud, this will re-fetch the file
 |              from cloud even if it is already found in a cache locally.
 |          override_config_path: path to a yaml config that will override the internal
 |              config file
 |          map_location: Optional torch.device() to map the instantiated model to a device.
 |              By default (None), it will select a GPU if available, falling back to CPU otherwise.
 |          strict: Passed to torch.load_state_dict. By default true.
 |          return_config: If set to true, will return just the underlying config of the restored
 |              model as an OmegaConf DictConfig object without instantiating the model.
 |          return_model_file: If set to true, will return just the downloaded model file in cache
 |
 |      Returns:
 |          A model instance of a particular model class or its underlying config (if return_config is set).
 |
 |  get_available_model_names() -> 'List[str]'
 |      Returns the list of model names available via NVIDIA NGC cloud,
 |      to get the complete model description use list_available_models()
 |      Returns:
 |          A list of model names
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.common.Serialization:
 |
 |  to_config_dict(self) -> "'DictConfig'"
 |      Returns object's configuration to config dictionary
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from nemo.core.classes.common.Serialization:
 |
 |  from_config_dict(config: "'DictConfig'", trainer: "Optional['Trainer']" = None)
 |      Instantiates object using DictConfig-based configuration
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.common.FileIO:
 |
 |  to_config_file(self, path2yaml_file: 'str')
 |      Saves current instance's configuration to YAML config file. Weights will not be saved.
 |      Args:
 |          path2yaml_file: path2yaml_file: path to yaml file where model model configuration will be saved
 |
 |      Returns:
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from nemo.core.classes.common.FileIO:
 |
 |  from_config_file(path2yaml_file: 'str')
 |      Instantiates an instance of NeMo Model from YAML config file.
 |      Weights will be initialized randomly.
 |      Args:
 |          path2yaml_file: path to yaml file with model configuration
 |
 |      Returns:
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO:
 |
 |  push_to_hf_hub(self, repo_id: str, *, pack_nemo_file: bool = True, model_card: Union[ForwardRef('ModelCard'), NoneType, object, str] = None, commit_message: str = 'Push model using huggingface_hub.', private: bool = False, api_endpoint: Optional[str] = None, token: Optional[str] = None, branch: Optional[str] = None, allow_patterns: Union[List[str], str, NoneType] = None, ignore_patterns: Union[List[str], str, NoneType] = None, delete_patterns: Union[List[str], str, NoneType] = None)
 |      Upload model checkpoint to the Hub.
 |
 |      Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
 |      `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
 |      details.
 |
 |      Args:
 |          repo_id (`str`):
 |              ID of the repository to push to (example: `"username/my-model"`).
 |          pack_nemo_file (`bool`, *optional*, defaults to `True`): Whether to pack the model checkpoint and
 |              configuration into a single `.nemo` file. If set to false, uploads the contents of the directory
 |              containing the model checkpoint and configuration plus additional artifacts.
 |          model_card (`ModelCard`, *optional*): Model card to upload with the model. If None, will use the model
 |              card template provided by the class itself via `generate_model_card()`. Any object that implements
 |              str(obj) can be passed here. Two keyword replacements are passed to `generate_model_card()`:
 |              `model_name` and `repo_id`. If the model card generates a string, and it contains `{model_name}` or
 |              `{repo_id}`, they will be replaced with the actual values.
 |          commit_message (`str`, *optional*):
 |              Message to commit while pushing.
 |          private (`bool`, *optional*, defaults to `False`):
 |              Whether the repository created should be private.
 |          api_endpoint (`str`, *optional*):
 |              The API endpoint to use when pushing the model to the hub.
 |          token (`str`, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. By default, it will use the token
 |              cached when running `huggingface-cli login`.
 |          branch (`str`, *optional*):
 |              The git branch on which to push the model. This defaults to `"main"`.
 |          allow_patterns (`List[str]` or `str`, *optional*):
 |              If provided, only files matching at least one pattern are pushed.
 |          ignore_patterns (`List[str]` or `str`, *optional*):
 |              If provided, files matching any of the patterns are not pushed.
 |          delete_patterns (`List[str]` or `str`, *optional*):
 |              If provided, remote files matching any of the patterns will be deleted from the repo.
 |
 |      Returns:
 |          The url of the uploaded HF repo.
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO:
 |
 |  get_hf_model_filter() -> Dict[str, Any]
 |      Generates a filter for HuggingFace models.
 |
 |      Additionaly includes default values of some metadata about results returned by the Hub.
 |
 |      Metadata:
 |          resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
 |          limit_results: Optional int, limits the number of results returned.
 |
 |      Returns:
 |          A dict representing the arguments passable to huggingface list_models().
 |
 |  search_huggingface_models(model_filter: Optional[Dict[str, Any]] = None) -> Iterable[ForwardRef('ModelInfo')]
 |      Should list all pre-trained models available via Hugging Face Hub.
 |
 |      The following metadata can be passed via the `model_filter` for additional results.
 |      Metadata:
 |
 |          resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
 |
 |          limit_results: Optional int, limits the number of results returned.
 |
 |      .. code-block:: python
 |
 |          # You can replace <DomainSubclass> with any subclass of ModelPT.
 |          from nemo.core import ModelPT
 |
 |          # Get default filter dict
 |          filt = <DomainSubclass>.get_hf_model_filter()
 |
 |          # Make any modifications to the filter as necessary
 |          filt['language'] = [...]
 |          filt['task'] = ...
 |          filt['tags'] = [...]
 |
 |          # Add any metadata to the filter as needed (kwargs to list_models)
 |          filt['limit'] = 5
 |
 |          # Obtain model info
 |          model_infos = <DomainSubclass>.search_huggingface_models(model_filter=filt)
 |
 |          # Browse through cards and select an appropriate one
 |          card = model_infos[0]
 |
 |          # Restore model using `modelId` of the card.
 |          model = ModelPT.from_pretrained(card.modelId)
 |
 |      Args:
 |          model_filter: Optional Dictionary (for Hugging Face Hub kwargs)
 |              that filters the returned list of compatible model cards, and selects all results from each filter.
 |              Users can then use `model_card.modelId` in `from_pretrained()` to restore a NeMo Model.
 |
 |      Returns:
 |          A list of ModelInfo entries.
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from nemo.collections.common.parts.optional_cuda_graphs.WithOptionalCudaGraphs:
 |
 |  disable_cuda_graphs_recursive(module: torch.nn.modules.module.Module, attribute_path: Optional[str] = None)
 |      Disable CUDA graphs Enable CUDA graphs, finding submodule recursively.
 |
 |      Args:
 |          module: instance of nn.Module
 |          attribute_path: field containing instance of WithOptionalCudaGraphs
 |                 E.g., "decoding.decoding" means that "<module>.decoding.decoding" are checked.
 |                 If None, "<module>" is checked.
 |
 |  enable_cuda_graphs_recursive(module: torch.nn.modules.module.Module, attribute_path: Optional[str] = None)
 |      Enable CUDA graphs, finding submodule recursively
 |
 |      Args:
 |          module: instance of nn.Module
 |          attribute_path: field containing instance of WithOptionalCudaGraphs
 |                 E.g., "decoding.decoding" means that "<module>.decoding.decoding" are checked.
 |                 If None, "<module>" is checked.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.collections.asr.parts.mixins.mixins.ASRModuleMixin:
 |
 |  change_attention_model(self, self_attention_model: str = None, att_context_size: List[int] = None, update_config: bool = True)
 |      Update the self_attention_model if function is available in encoder.
 |
 |      Args:
 |          self_attention_model (str): type of the attention layer and positional encoding
 |
 |              'rel_pos':
 |                  relative positional embedding and Transformer-XL
 |
 |              'rel_pos_local_attn':
 |                  relative positional embedding and Transformer-XL with local attention using
 |                  overlapping windows. Attention context is determined by att_context_size parameter.
 |
 |              'abs_pos':
 |                  absolute positional embedding and Transformer
 |
 |              If None is provided, the self_attention_model isn't changed. Defauts to None.
 |          att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes,
 |              or None to keep as it is. Defauts to None.
 |          update_config (bool): Whether to update the config or not with the new attention model.
 |              Defaults to True.
 |
 |  change_conv_asr_se_context_window(self, context_window: int, update_config: bool = True)
 |      Update the context window of the SqueezeExcitation module if the provided model contains an
 |      `encoder` which is an instance of `ConvASREncoder`.
 |
 |      Args:
 |          context_window:  An integer representing the number of input timeframes that will be used
 |              to compute the context. Each timeframe corresponds to a single window stride of the
 |              STFT features.
 |
 |              Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
 |              of context to compute the Squeeze step.
 |          update_config: Whether to update the config or not with the new context window.
 |
 |  change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int, update_config: bool = True)
 |      Update the conv_chunking_factor (int) if function is available in encoder.
 |      Default is 1 (auto)
 |      Set it to -1 (disabled) or to a specific value (power of 2) if you OOM in the conv subsampling layers
 |
 |      Args:
 |          conv_chunking_factor (int)
 |
 |  conformer_stream_step(self, processed_signal: torch.Tensor, processed_signal_length: torch.Tensor = None, cache_last_channel: torch.Tensor = None, cache_last_time: torch.Tensor = None, cache_last_channel_len: torch.Tensor = None, keep_all_outputs: bool = True, previous_hypotheses: List[nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis] = None, previous_pred_out: torch.Tensor = None, drop_extra_pre_encoded: int = None, return_transcription: bool = True, return_log_probs: bool = False, bypass_pre_encode: bool = False)
 |      It simulates a forward step with caching for streaming purposes.
 |      It supports the ASR models where their encoder supports streaming like Conformer.
 |      Args:
 |          processed_signal: the input audio signals
 |          processed_signal_length: the length of the audios
 |          cache_last_channel: the cache tensor for last channel layers like MHA
 |          cache_last_channel_len: lengths for cache_last_channel
 |          cache_last_time: the cache tensor for last time layers like convolutions
 |          keep_all_outputs: if set to True, would not drop the extra outputs specified by encoder.streaming_cfg.valid_out_len
 |          previous_hypotheses: the hypotheses from the previous step for RNNT models
 |          previous_pred_out: the predicted outputs from the previous step for CTC models
 |          drop_extra_pre_encoded: number of steps to drop from the beginning of the outputs after the downsampling module. This can be used if extra paddings are added on the left side of the input.
 |          return_transcription: whether to decode and return the transcriptions. It can not get disabled for Transducer models.
 |          return_log_probs: whether to return the log probs, only valid for ctc model
 |
 |      Returns:
 |          greedy_predictions: the greedy predictions from the decoder
 |          all_hyp_or_transcribed_texts: the decoder hypotheses for Transducer models and the transcriptions for CTC models
 |          cache_last_channel_next: the updated tensor cache for last channel layers to be used for next streaming step
 |          cache_last_time_next: the updated tensor cache for last time layers to be used for next streaming step
 |          cache_last_channel_next_len: the updated lengths for cache_last_channel
 |          best_hyp: the best hypotheses for the Transducer models
 |          log_probs: the logits tensor of current streaming chunk, only returned when return_log_probs=True
 |          encoded_len: the length of the output log_probs + history chunk log_probs, only returned when return_log_probs=True
 |
 |  transcribe_simulate_cache_aware_streaming(self, paths2audio_files: List[str], batch_size: int = 4, logprobs: bool = False, return_hypotheses: bool = False, online_normalization: bool = False)
 |      Args:
 |          paths2audio_files: (a list) of paths to audio files.
 |          batch_size: (int) batch size to use during inference.
 |              Bigger will result in better throughput performance but would use more memory.
 |          logprobs: (bool) pass True to get log probabilities instead of transcripts.
 |          return_hypotheses: (bool) Either return hypotheses or text
 |              With hypotheses can do some postprocessing like getting timestamp or rescoring
 |          online_normalization: (bool) Perform normalization on the run per chunk.
 |      Returns:
 |          A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.collections.asr.parts.mixins.asr_adapter_mixins.ASRAdapterModelMixin:
 |
 |  add_adapter(self, name: str, cfg: omegaconf.dictconfig.DictConfig)
 |      Add an Adapter module to this model.
 |
 |      Args:
 |          name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
 |          cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Adapter module.
 |
 |  check_valid_model_with_adapter_support_(self)
 |      Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.
 |
 |  get_enabled_adapters(self) -> List[str]
 |      Returns a list of all enabled adapters.
 |
 |      Returns:
 |          A list of str names of each enabled adapter(s).
 |
 |  is_adapter_available(self) -> bool
 |      Checks if any Adapter module has been instantiated.
 |
 |      Returns:
 |          bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
 |          enabled or disabled, false only if no adapters exist.
 |
 |  resolve_adapter_module_name_(self, name: str) -> Tuple[str, str]
 |      Utility method to resolve a given global/module adapter name to its components.
 |      Always returns a tuple representing (module_name, adapter_name). ":" is used as the
 |      delimiter for denoting the module name vs the adapter name.
 |
 |      Will attempt to also resolve a given adapter_name alone back to (module_name, adapter_name)
 |      if the metadata config exists for access.
 |
 |      Args:
 |          name: A global adapter, or a module adapter name (with structure module_name:adapter_name).
 |
 |      Returns:
 |          A tuple representing (module_name, adapter_name). If a global adapter is provided,
 |          module_name is set to ''.
 |
 |  set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True)
 |      Updated the internal adapter config, determining if an adapter (or all adapters) are either
 |      enabled or disabled.
 |
 |      A common user pattern would be to disable all adapters (either after adding them, or restoring a model
 |      with pre-existing adapters) and then simply enable one of the adapters.
 |
 |      .. code::
 |
 |          model.set_enabled_adapters(enabled=False)
 |          model.set_enabled_adapters(name=<some adapter name>, enabled=True)
 |
 |      Args:
 |          name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
 |              If no name is given, then all adapters will be enabled/disabled.
 |          enabled: Bool, determines if the adapter(s) will be enabled/disabled.
 |
 |  setup_adapters(self)
 |      Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
 |      adapters that were previously added.
 |
 |      This method should be called just once at constructor time.
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from nemo.collections.asr.parts.mixins.asr_adapter_mixins.ASRAdapterModelMixin:
 |
 |  adapter_module_names
 |      List of valid adapter modules that are supported by the model.
 |
 |      .. note::
 |
 |          Subclasses should override this property and return a list of str names, of all the modules
 |          that they support, which will enable users to determine where to place the adapter modules.
 |
 |      Returns:
 |          A list of str, one for each of the adapter modules that are supported. By default, the subclass
 |          should support the "default adapter" ('').
 |
 |  default_adapter_module_name
 |      Name of the adapter module that is used as "default" if a name of '' is provided.
 |
 |      .. note::
 |
 |          Subclasses should override this property and return a str name of the module
 |          that they wish to denote as the default.
 |
 |      Returns:
 |          A str name of a module, which is denoted as 'default' adapter or None. If None, then no default
 |          adapter is supported.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.mixins.adapter_mixins.AdapterModelPTMixin:
 |
 |  load_adapters(self, filepath: str, name: str = None, map_location: str = None, strict: bool = True)
 |      Utility method that restores only the adapter module(s), and not the entire model itself.
 |      This allows the sharing of adapters which are often just a fraction of the size of the full model,
 |      enabling easier deliver.
 |
 |      .. note::
 |
 |          During restoration, assumes that the model does not currently already have an adapter with
 |          the name (if provided), or any adapter that shares a name with the state dict's modules
 |          (if name is not provided). This is to ensure that each adapter name is globally unique
 |          in a model.
 |
 |      Args:
 |          filepath: Filepath of the .pt file.
 |          name: Optional name of the adapter that will be saved to this file. If None is passed,
 |              all adapters will be saved to the file. The name must be either the global name (adapter_name),
 |              or the module level name (module:adapter_name), whichever exactly matches the state dict.
 |          map_location: Pytorch flag, where to place the adapter(s) state dict(s).
 |          strict: Pytorch flag, whether to load the weights of the adapter(s) strictly or not.
 |
 |  replace_adapter_compatible_modules(self, update_config: bool = True, verbose: bool = True)
 |      Utility method to replace all child modules with Adapter variants, if they exist.
 |      Does NOT recurse through children of children modules (only immediate children).
 |
 |      Args:
 |          update_config: A flag that determines if the config should be updated or not.
 |          verbose: A flag that determines if the method should log the changes made or not.
 |
 |  save_adapters(self, filepath: str, name: str = None)
 |      Utility method that saves only the adapter module(s), and not the entire model itself.
 |      This allows the sharing of adapters which are often just a fraction of the size of the full model,
 |      enabling easier deliver.
 |
 |      .. note::
 |
 |          The saved file is a pytorch compatible pickle file, containing the state dicts of the adapter(s),
 |          as well as a binary representation of the adapter config.
 |
 |      Args:
 |          filepath: A str filepath where the .pt file that will contain the adapter state dict.
 |          name: Optional name of the adapter that will be saved to this file. If None is passed,
 |              all adapters will be saved to the file. The name can be either the global name (adapter_name),
 |              or the module level name (module:adapter_name).
 |
 |  update_adapter_cfg(self, cfg: omegaconf.dictconfig.DictConfig)
 |      Utility method to recursively update all of the Adapter module configs with the provided config.
 |
 |      .. note::
 |
 |          It is not a (deep)copy, but a reference copy. Changes made to the config will be reflected to
 |          adapter submodules, but it is still encouraged to explicitly update the adapter_cfg using this method.
 |
 |      Args:
 |          cfg: DictConfig containing the value of `model.cfg.adapters`.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.mixins.adapter_mixins.AdapterModuleMixin:
 |
 |  check_supported_adapter_type_(self, adapter_cfg: omegaconf.dictconfig.DictConfig, supported_adapter_types: Optional[Iterable[type]] = None)
 |      Utility method to check if the adapter module is a supported type by the module.
 |
 |      This method should be called by the subclass to ensure that the adapter module is a supported type.
 |
 |  forward_enabled_adapters(self, input: 'torch.Tensor')
 |      Forward's all active adapters one by one with the provided input, and chaining the outputs of each
 |      adapter layer to the next.
 |
 |      Utilizes the implicit merge strategy of each adapter when computing the adapter's output, and
 |      how that output will be merged back with the original input.
 |
 |      Args:
 |          input: The output tensor of the calling module is the input to the first adapter, whose output
 |              is then chained to the next adapter until all adapters are consumed.
 |
 |      Returns:
 |          The result tensor, after all active adapters have finished their forward passes.
 |
 |  forward_single_enabled_adapter_(self, input: torch.Tensor, adapter_module: torch.nn.modules.module.Module, *, adapter_name: str, adapter_strategy: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy')
 |      Perform the forward step of a single adapter module on some input data.
 |
 |      .. note::
 |
 |          Subclasses can override this method to accommodate more complicate adapter forward steps.
 |
 |      Args:
 |          input: input: The output tensor of the calling module is the input to the first adapter, whose output
 |              is then chained to the next adapter until all adapters are consumed.
 |          adapter_module: The adapter module that is currently required to perform the forward pass.
 |          adapter_name: The resolved name of the adapter that is undergoing the current forward pass.
 |          adapter_strategy: A subclass of `AbstractAdapterStrategy`, that determines how the
 |              output of the adapter should be merged with the input, or if it should be merged at all.
 |
 |      Returns:
 |          The result tensor, after the current active adapter has finished its forward pass.
 |
 |  get_accepted_adapter_types(self) -> Set[type]
 |      Utility function to get the set of all classes that are accepted by the module.
 |
 |      Returns:
 |          Returns the set of accepted adapter types as classes, otherwise an empty set.
 |
 |  get_adapter_cfg(self, name: str)
 |      Same logic as `get_adapter_module` but to get the config
 |
 |  get_adapter_module(self, name: str)
 |      Gets an adapter module by name if possible, otherwise returns None.
 |
 |      Args:
 |          name: A str name (resolved or not) corresponding to an Adapter.
 |
 |      Returns:
 |          An nn.Module if the name could be resolved and matched, otherwise None/
 |
 |  set_accepted_adapter_types(self, adapter_types: List[Union[type, str]]) -> None
 |      The module with this mixin can define a list of adapter names that it will accept.
 |      This method should be called in the modules init method and set the adapter names
 |      the module will expect to be added.
 |
 |      Args:
 |          adapter_types: A list of str paths that correspond to classes. The class paths will be instantiated to
 |              ensure that the class path is correct.
 |
 |  unfreeze_enabled_adapters(self, freeze_batchnorm: bool = True) -> None
 |      Utility method to unfreeze only the enabled Adapter module(s).
 |
 |      A common user pattern is to freeze all the modules (including all the adapters), and then
 |      unfreeze just the required adapters.
 |
 |      .. code::
 |
 |          module.freeze()  # only available to nemo.core.NeuralModule !
 |          module.unfreeze_enabled_adapters()
 |
 |      Args:
 |          freeze_batchnorm: An optional (and recommended) practice of freezing the updates to the moving average
 |              buffers of any and all BatchNorm*D layers. This is necessary to ensure that disabling all adapters
 |              will precisely yield the original (base) model's outputs.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from nemo.core.classes.mixins.adapter_mixins.AdapterModuleMixin:
 |
 |  adapter_global_cfg_key = 'global_cfg'
 |
 |  adapter_metadata_cfg_key = 'adapter_meta_cfg'
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.collections.asr.models.asr_model.ExportableEncDecModel:
 |
 |  forward_for_export(self, audio_signal, length=None, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None)
 |      This forward is used when we need to export the model to ONNX format.
 |      Inputs cache_last_channel and cache_last_time are needed to be passed for exporting streaming models.
 |
 |      Args:
 |          input: Tensor that represents a batch of raw audio signals of shape [B, T]. T here represents timesteps.
 |          length: Vector of length B, that contains the individual lengths of the audio sequences.
 |          cache_last_channel: Tensor of shape [N, B, T, H] which contains the cache for last channel layers
 |          cache_last_time: Tensor of shape [N, B, H, T] which contains the cache for last time layers
 |              N is the number of such layers which need caching, B is batch size, H is the hidden size of activations,
 |              and T is the length of the cache
 |
 |      Returns:
 |          the output of the model
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from nemo.collections.asr.models.asr_model.ExportableEncDecModel:
 |
 |  disabled_deployment_input_names
 |      Implement this method to return a set of input names disabled for export
 |
 |  disabled_deployment_output_names
 |      Implement this method to return a set of output names disabled for export
 |
 |  input_module
 |
 |  output_module
 |
 |  output_names
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.core.classes.exportable.Exportable:
 |
 |  dynamic_shapes_for_export(self, use_dynamo=False)
 |
 |  export(self, output: str, input_example=None, verbose=False, do_constant_folding=True, onnx_opset_version=None, check_trace: Union[bool, List[torch.Tensor]] = False, dynamic_axes=None, check_tolerance=0.01, export_modules_as_functions=False, keep_initializers_as_inputs=None, use_dynamo=False)
 |      Exports the model to the specified format. The format is inferred from the file extension of the output file.
 |
 |      Args:
 |          output (str): Output file name. File extension be .onnx, .pt, or .ts, and is used to select export
 |              path of the model.
 |          input_example (list or dict): Example input to the model's forward function. This is used to
 |              trace the model and export it to ONNX/TorchScript. If the model takes multiple inputs, then input_example
 |              should be a list of input examples. If the model takes named inputs, then input_example
 |              should be a dictionary of input examples.
 |          verbose (bool): If True, will print out a detailed description of the model's export steps, along with
 |              the internal trace logs of the export process.
 |          do_constant_folding (bool): If True, will execute constant folding optimization on the model's graph
 |              before exporting. This is ONNX specific.
 |          onnx_opset_version (int): The ONNX opset version to export the model to. If None, will use a reasonable
 |              default version.
 |          check_trace (bool): If True, will verify that the model's output matches the output of the traced
 |              model, upto some tolerance.
 |          dynamic_axes (dict): A dictionary mapping input and output names to their dynamic axes. This is
 |              used to specify the dynamic axes of the model's inputs and outputs. If the model takes multiple inputs,
 |              then dynamic_axes should be a list of dictionaries. If the model takes named inputs, then dynamic_axes
 |              should be a dictionary of dictionaries. If None, will use the dynamic axes of the input_example
 |              derived from the NeuralType of the input and output of the model.
 |          check_tolerance (float): The tolerance to use when checking the model's output against the traced
 |              model's output. This is only used if check_trace is True. Note the high tolerance is used because
 |              the traced model is not guaranteed to be 100% accurate.
 |          export_modules_as_functions (bool): If True, will export the model's submodules as functions. This is
 |              ONNX specific.
 |          keep_initializers_as_inputs (bool): If True, will keep the model's initializers as inputs in the onnx graph.
 |              This is ONNX specific.
 |          use_dynamo (bool): If True, use onnx.dynamo_export() instead of onnx.export(). This is ONNX specific.
 |
 |      Returns:
 |          A tuple of two outputs.
 |          Item 0 in the output is a list of outputs, the outputs of each subnet exported.
 |          Item 1 in the output is a list of string descriptions. The description of each subnet exported can be
 |          used for logging purposes.
 |
 |  get_export_config(self)
 |      Returns export_config dictionary
 |
 |  get_export_subnet(self, subnet=None)
 |      Returns Exportable subnet model/module to export
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from nemo.core.classes.exportable.Exportable:
 |
 |  input_names
 |
 |  input_types_for_export
 |
 |  output_types_for_export
 |
 |  supported_export_formats
 |      Implement this method to return a set of export formats supported. Default is all types.
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from nemo.collections.asr.parts.mixins.transcription.ASRTranscriptionMixin:
 |
 |  get_transcribe_config() -> nemo.collections.asr.parts.mixins.transcription.TranscribeConfig
 |      Utility method that returns the default config for transcribe() function.
 |
 |      Returns:
 |          A dataclass
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.collections.asr.parts.mixins.transcription.TranscriptionMixin:
 |
 |  transcribe_generator(self, audio, override_config: Optional[nemo.collections.asr.parts.mixins.transcription.TranscribeConfig])
 |      A generator version of `transcribe` function.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from nemo.collections.asr.parts.mixins.mixins.ASRBPEMixin:
 |
 |  save_tokenizers(self, directory: str)
 |      Save the model tokenizer(s) to the specified directory.
 |
 |      Args:
 |          directory: The directory to save the tokenizer(s) to.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from nemo.collections.asr.parts.mixins.mixins.ASRBPEMixin:
 |
 |  AGGREGATE_TOKENIZERS_DICT_PREFIX = 'langs'