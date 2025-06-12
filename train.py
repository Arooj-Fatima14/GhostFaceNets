# import os
# import data
# import evals
# import losses
# import GhostFaceNets, GhostFaceNets_with_Bias
# import myCallbacks
# import tensorflow as tf
# from tensorflow import keras
# import models

# # import multiprocessing as mp

# # if mp.get_start_method() != "forkserver":
# #     mp.set_start_method("forkserver", force=True)

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# # strategy = tf.distribute.MirroredStrategy()
# # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")


# class Train:
#     def __init__(
#         self,
#         data_path,
#         save_path,
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,  # L2 regularizer for output layer, 0 for None, >=1 for value in basic_model, (0, 1) for specific value
#         custom_objects={},
#         pretrained=None,  # If reload weights from another h5 file
#         batch_size=128,
#         lr_base=0.001,
#         lr_decay=0.05,  # for cosine it's m_mul, or it's decay_rate for exponential or constant
#         lr_decay_steps=0,  # <=1 for Exponential, (1, 500) for Cosine decay on epoch, >= 500 for Cosine decay on batch, list for Constant
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,  # ratio of randomly cutout bottom 2/5 area, regarding as ignoring mask area
#         image_per_class=0,  # For triplet, image_per_class will be `4` if it's `< 4`
#         samples_per_mining=0,  # **Not working well**. Set a value > 0 will use offline_triplet_mining dataset
#         mixup_alpha=0,  # mixup alpha, value in (0, 1] to enable
#         partial_fc_split=0,  # **Not working well**. Set a int number like `2`, will build model and dataset with total classes split in parts.
#         teacher_model_interf=None,  # Teacher model to generate embedding data, used for distilling training.
#         sam_rho=0,
#         vpl_start_iters=-1,  # Enable by setting value > 0, like 8000. https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Variational_Prototype_Learning_for_Deep_Face_Recognition_CVPR_2021_paper.pdf
#         vpl_allowed_delta=200,
#     ):
#         from inspect import getmembers, isfunction, isclass

#         custom_objects.update(dict([ii for ii in getmembers(losses) if isfunction(ii[1]) or isclass(ii[1])]))
#         custom_objects.update({"NormDense": models.NormDense})

#         self.model, self.basic_model, self.save_path, self.inited_from_model, self.sam_rho, self.pretrained = None, None, save_path, False, sam_rho, pretrained
#         self.vpl_start_iters, self.vpl_allowed_delta = vpl_start_iters, vpl_allowed_delta
        
#         #if model is not specified, try to reload model from checkpoints
#         if model is None and basic_model is None:
#             model = os.path.join("checkpoints", save_path)
#             print(">>>> Try reload from:", model)

#         #If model passed is string
#         if isinstance(model, str):
#             if model.endswith(".h5") and os.path.exists(model):
#                 print(">>>> Load model from h5 file: %s..." % model)
#                 with keras.utils.custom_object_scope(custom_objects):
#                     self.model = keras.models.load_model(model, compile=compile, custom_objects=custom_objects)
#                 embedding_layer = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
#                 self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[embedding_layer].output)
#                 # self.model.summary()

#         #If model passed is keras.models.Model
#         elif isinstance(model, keras.models.Model):
#             self.model = model
#             embedding_layer = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
#             self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[embedding_layer].output)
#             self.inited_from_model = True
#             print(">>>> Specified model structure, output layer will keep from changing")
        
#         #If basic model is string
#         elif isinstance(basic_model, str):
#             if basic_model.endswith(".h5") and os.path.exists(basic_model):
#                 print(">>>> Load basic_model from h5 file: %s..." % basic_model)
#                 with keras.utils.custom_object_scope(custom_objects):
#                     self.basic_model = keras.models.load_model(basic_model, compile=compile, custom_objects=custom_objects)
        
#         #If basic model passed is keras.models.Model
#         elif isinstance(basic_model, keras.models.Model):
#             self.basic_model = basic_model

#         if self.basic_model == None:
#             print(
#                 "Initialize model by:\n"
#                 "| basic_model                                                     | model           |\n"
#                 "| --------------------------------------------------------------- | --------------- |\n"
#                 "| model structure                                                 | None            |\n"
#                 "| basic model .h5 file                                            | None            |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
#                 "| None                                                            | None            |\n"
#                 "* Both None for reload model from 'checkpoints/{}'\n".format(save_path)
#             )
#             return

#         #Losses
#         self.softmax, self.arcface, self.arcface_partial, self.triplet = "softmax", "arcface", "arcface_partial", "triplet"
#         self.center, self.distill = "center", "distill"
        

#         if output_weight_decay >= 1:
#             l2_weight_decay = 0
#             for ii in self.basic_model.layers:
#                 #Look for "reqularizer" in model's layers
#                 if hasattr(ii, "kernel_regularizer") and isinstance(ii.kernel_regularizer, keras.regularizers.L2):
#                     l2_weight_decay = ii.kernel_regularizer.l2
#                     break
#             print(">>>> L2 regularizer value from basic_model:", l2_weight_decay)
#             output_weight_decay *= l2_weight_decay * 2
#         self.output_weight_decay = output_weight_decay

#         self.batch_size, self.batch_size_per_replica = batch_size, batch_size
#         if tf.distribute.has_strategy():
#             strategy = tf.distribute.get_strategy()
#             self.batch_size = batch_size * strategy.num_replicas_in_sync
#             print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
#             self.data_options = tf.data.Options()
#             self.data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
#         #Evaluate
#         my_evals = [evals.eval_callback(self.basic_model, ii, batch_size=self.batch_size_per_replica, eval_freq=eval_freq) for ii in eval_paths]
#         if len(my_evals) != 0:
#             my_evals[-1].save_model = os.path.splitext(save_path)[0]
        
#         #
#         self.my_history, self.model_checkpoint, self.lr_scheduler, self.gently_stop = myCallbacks.basic_callbacks(
#             save_path,
#             my_evals,
#             lr=lr_base,
#             lr_decay=lr_decay,
#             lr_min=lr_min,
#             lr_decay_steps=lr_decay_steps,
#             lr_warmup_steps=lr_warmup_steps,
#         )
#         self.gently_stop = None  # may not working for windows
#         self.my_evals, self.custom_callbacks = my_evals, []
#         self.metrics = ["accuracy"]
#         self.default_optimizer = "adam"

#         self.data_path, self.random_status, self.image_per_class, self.mixup_alpha = data_path, random_status, image_per_class, mixup_alpha
#         self.random_cutout_mask_area, self.partial_fc_split, self.samples_per_mining = random_cutout_mask_area, partial_fc_split, samples_per_mining
#         self.train_ds, self.steps_per_epoch, self.classes, self.is_triplet_dataset = None, None, 0, False
#         self.teacher_model_interf, self.is_distill_ds = teacher_model_interf, False
#         self.distill_emb_map_layer = None

#     #Get embedding layer
#     def __search_embedding_layer__(self, model):
#         for ii in range(1, 6):
#             if model.layers[-ii].name == "embedding":
#                 return -ii
    
#     #Initialize the dataset
#     def __init_dataset__(self, type, emb_loss_names):
#         init_as_triplet = self.triplet in emb_loss_names or type == self.triplet
#         is_offline_triplet = self.samples_per_mining > 0
#         if self.train_ds is not None and init_as_triplet == self.is_triplet_dataset and not self.is_distill_ds and not is_offline_triplet:
#             return

#         dataset_params = {
#             "data_path": self.data_path,
#             "batch_size": self.batch_size,
#             "random_status": self.random_status,
#             "random_cutout_mask_area": self.random_cutout_mask_area,
#             "image_per_class": self.image_per_class,
#             "mixup_alpha": self.mixup_alpha,
#             "teacher_model_interf": self.teacher_model_interf,
#         }

#         if is_offline_triplet:
#             print(">>>> Init offline triplet dataset...")
#             aa = data.Triplet_dataset_offline(basic_model=self.basic_model, samples_per_mining=self.samples_per_mining, **dataset_params)
#             self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = False
#         elif init_as_triplet:
#             print(">>>> Init triplet dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 print(">>>> Combining tfrecord dataset with triplet is NOT recommended.")
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 aa = data.Triplet_dataset(**dataset_params)
#                 self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = True
#         else:
#             print(">>>> Init softmax dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 self.train_ds, self.steps_per_epoch = data.prepare_dataset(**dataset_params, partial_fc_split=self.partial_fc_split)
#             self.is_triplet_dataset = False
#         if self.train_ds is None:
#             return

#         if tf.distribute.has_strategy():
#             self.train_ds = self.train_ds.with_options(self.data_options)

#         label_spec = self.train_ds.element_spec[-1]
#         if isinstance(label_spec, tuple):
#             # dataset with embedding values
#             self.is_distill_ds = True
#             self.teacher_emb_size = label_spec[0].shape[-1]
#             self.classes = label_spec[1].shape[-1]
#             if type == self.distill:
#                 # Loss is distill type: [label * n, embedding]
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[1:] * len(emb_loss_names) + yy[:1]))
#             elif (self.distill in emb_loss_names and len(emb_loss_names) != 1) or (self.distill not in emb_loss_names and len(emb_loss_names) != 0):
#                 # Will attach distill loss as embedding loss, and there are other embedding losses: [embedding, label * n]
#                 label_data_len = len(emb_loss_names) if self.distill in emb_loss_names else len(emb_loss_names) + 1
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[:1] + yy[1:] * label_data_len))
#         else:
#             self.is_distill_ds = False
#             self.classes = label_spec.shape[-1]
    
#     #Determine the optimizer
#     def __init_optimizer__(self, optimizer):
#         if optimizer == None:
#             if self.model != None and self.model.optimizer != None:
#                 # Model loaded from .h5 file already compiled
#                 # Saving may meet Error: OSError: Unable to create link (name already exists)
#                 self.optimizer = self.model.optimizer
#                 compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#                 print(">>>> Reuse optimizer from previoue model:", compiled_opt.__class__.__name__)
#                 # if isinstance(self.model.optimizer, keras.mixed_precision.LossScaleOptimizer):
#                 #     inner_optimizer_pre = self.model.optimizer.inner_optimizer
#                 #     inner_optimizer = inner_optimizer_pre.__class__(**inner_optimizer_pre.get_config())
#                 #     # self.optimizer = keras.mixed_precision.LossScaleOptimizer(inner_optimizer)
#                 #     self.optimizer = inner_optimizer
#                 # else:
#                 #     self.optimizer = self.model.optimizer.__class__(**self.model.optimizer.get_config())
#             else:
#                 print(">>>> Use default optimizer:", self.default_optimizer)
#                 self.optimizer = self.default_optimizer
#         else:
#             print(">>>> Use specified optimizer:", optimizer)
#             self.optimizer = optimizer

#         try:
#             import tensorflow_addons as tfa
#         except:
#             pass
#         else:
#             compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#             if isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
#                 print(">>>> Append weight decay callback...")
#                 lr_base, wd_base = self.optimizer.lr.numpy(), self.optimizer.weight_decay.numpy()
#                 wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=self.is_lr_on_batch)
#                 self.callbacks.append(wd_callback)  # should be after lr_scheduler

#     def __init_model__(self, type, loss_top_k=1, header_append_norm=False):
#         inputs = self.basic_model.inputs[0]
#         embedding = self.basic_model.outputs[0]
#         is_multi_output = lambda mm: len(mm.outputs) != 1 or isinstance(mm.layers[-1], keras.layers.Concatenate)
#         if self.model != None and is_multi_output(self.model):
#             output_layer = min(len(self.basic_model.layers), len(self.model.layers) - 1)
#             self.model = keras.models.Model(inputs, self.model.layers[output_layer].output)

#         #Add regularization on model's output
#         if self.output_weight_decay != 0:
#             print(">>>> Add L2 regularizer to model output layer, output_weight_decay = %f" % self.output_weight_decay)
#             output_kernel_regularizer = keras.regularizers.L2(self.output_weight_decay / 2)
#         else:
#             output_kernel_regularizer = None

#         #
#         model_output_layer_name = None if self.model is None else self.model.output_names[-1]
#         # arcface_not_match = self.model.layers[-1].append_norm != header_append_norm or self.partial_fc_split != self.model.layers[-1].partial_fc_split
#         if type == self.softmax and model_output_layer_name != self.softmax:
#             print(">>>> Add softmax layer...")
#             softmax_logits = keras.layers.Dense(self.classes, use_bias=False, name=self.softmax + "_logits", kernel_regularizer=output_kernel_regularizer)
#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 softmax_logits.build(embedding.shape)
#                 weight_cur = softmax_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     softmax_logits.set_weights(weight_pre)
#             logits = softmax_logits(embedding)
#             output_fp32 = keras.layers.Activation("softmax", dtype="float32", name=self.softmax)(logits)
#             self.model = keras.models.Model(inputs, output_fp32)
#         elif type == self.arcface and (model_output_layer_name != self.arcface or self.model.layers[-1].append_norm != header_append_norm):
#             vpl_start_iters = self.vpl_start_iters * self.steps_per_epoch if self.vpl_start_iters < 50 else self.vpl_start_iters
#             vpl_kwargs = {"vpl_lambda": 0.15, "start_iters": vpl_start_iters, "allowed_delta": self.vpl_allowed_delta}
#             arc_kwargs = {"loss_top_k": loss_top_k, "append_norm": header_append_norm, "partial_fc_split": self.partial_fc_split, "name": self.arcface}
#             print(">>>> Add arcface layer, arc_kwargs={}, vpl_kwargs={}...".format(arc_kwargs, vpl_kwargs))
#             if vpl_start_iters > 0:
#                 batch_size = self.batch_size_per_replica
#                 arcface_logits = models.NormDenseVPL(batch_size, self.classes, output_kernel_regularizer, **arc_kwargs, **vpl_kwargs, dtype="float32")
#             else:
#                 arcface_logits = models.NormDense(self.classes, output_kernel_regularizer, **arc_kwargs, dtype="float32")

#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 arcface_logits.build(embedding.shape)
#                 weight_cur = arcface_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     arcface_logits.set_weights(weight_pre)
#             output_fp32 = arcface_logits(embedding)
#             # output_fp32 = keras.layers.Activation('linear', dtype='float32', name=self.arcface)(output_fp32)
#             self.model = keras.models.Model(inputs, output_fp32)
#         elif type in [self.triplet, self.center, self.distill]:
#             self.model = self.basic_model
#             self.model.output_names[0] = type + "_embedding"
#         else:
#             print(">>>> Will NOT change model output layer.")

#         if self.pretrained is not None:
#             if self.model is None:
#                 self.basic_model.load_weights(self.pretrained)
#             else:
#                 self.model.load_weights(self.pretrained)
#             self.pretrained = None

#     def __add_emb_output_to_model__(self, emb_type, emb_loss, emb_loss_weight):
#         nns = self.model.output_names
#         emb_shape = self.basic_model.output_shape[-1]
#         if emb_type == self.distill and self.teacher_emb_size != emb_shape:
#             print(">>>> Add a dense layer to map embedding: student %d --> teacher %d" % (emb_shape, self.teacher_emb_size))
#             embedding = self.basic_model.outputs[0]
#             if self.distill_emb_map_layer is None:
#                 self.distill_emb_map_layer = keras.layers.Dense(self.teacher_emb_size, use_bias=False, name="distill_map", dtype="float32")
#             emb_map_output = self.distill_emb_map_layer(embedding)
#             self.model = keras.models.Model(self.model.inputs[0], [emb_map_output] + self.model.outputs)
#         else:
#             self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)

#         self.model.output_names[0] = emb_type + "_embedding"
#         for id, nn in enumerate(nns):
#             self.model.output_names[id + 1] = nn
#         self.cur_loss = [emb_loss, *self.cur_loss]
#         self.loss_weights.update({self.model.output_names[0]: emb_loss_weight})

#     def __init_type_by_loss__(self, loss):
#         print(">>>> Init type by loss function name...")
#         if isinstance(loss, str):
#             return self.softmax

#         if loss.__class__.__name__ == "function":
#             ss = loss.__name__.lower()
#             if self.softmax in ss:
#                 return self.softmax
#             if self.arcface in ss:
#                 return self.arcface
#             if self.triplet in ss:
#                 return self.triplet
#             if self.distill in ss:
#                 return self.distill
#         else:
#             ss = loss.__class__.__name__.lower()
#             if isinstance(loss, losses.TripletLossWapper) or self.triplet in ss:
#                 return self.triplet
#             if isinstance(loss, losses.CenterLoss) or self.center in ss:
#                 return self.center
#             if isinstance(loss, losses.ArcfaceLoss) or self.arcface in ss:
#                 return self.arcface
#             if isinstance(loss, losses.ArcfaceLossSimple) or isinstance(loss, losses.AdaCosLoss):
#                 return self.arcface
#             if isinstance(loss, losses.DistillKLDivergenceLoss):
#                 return self.arcface  # Use NormDense layer
#             if self.softmax in ss:
#                 return self.softmax
#         return self.softmax

#     def __init_emb_losses__(self, embLossTypes=None, embLossWeights=1):
#         emb_loss_names, emb_loss_weights = {}, {}
#         if embLossTypes is not None:
#             embLossTypes = embLossTypes if isinstance(embLossTypes, list) else [embLossTypes]
#             for id, ee in enumerate(embLossTypes):
#                 emb_loss_name = ee.lower() if isinstance(ee, str) else ee.__name__.lower()
#                 emb_loss_weight = float(embLossWeights[id] if isinstance(embLossWeights, list) else embLossWeights)
#                 if "centerloss" in emb_loss_name:
#                     emb_loss_names[self.center] = losses.CenterLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.center] = emb_loss_weight
#                 elif "triplet" in emb_loss_name:
#                     emb_loss_names[self.triplet] = losses.BatchHardTripletLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.triplet] = emb_loss_weight
#                 elif "distill" in emb_loss_name:
#                     emb_loss_names[self.distill] = losses.distiller_loss_cosine if ee == None or isinstance(ee, str) else ee
#                     emb_loss_weights[self.distill] = emb_loss_weight
#         return emb_loss_names, emb_loss_weights

#     def __basic_train__(self, epochs, initial_epoch=0):
#         self.model.compile(optimizer=self.optimizer, loss=self.cur_loss, metrics=self.metrics, loss_weights=self.loss_weights)
#         self.model.fit(
#             self.train_ds,
#             epochs=epochs,
#             verbose=1,
#             callbacks=self.callbacks,
#             initial_epoch=initial_epoch,
#             steps_per_epoch=self.steps_per_epoch,
#             # steps_per_epoch=0,
#             use_multiprocessing=True,
#             workers=4,
#         )

#     def reset_dataset(self, data_path=None):
#         self.train_ds = None
#         if data_path != None:
#             self.data_path = data_path

#     def train_single_scheduler(
#         self,
#         epoch,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35,
#     ):
#         emb_loss_names, emb_loss_weights = self.__init_emb_losses__(embLossTypes, embLossWeights)

#         if loss is None:
#             if self.model.built:
#                 loss = self.model.loss[0]
#             else:
#                 return

#         if type is None and not self.inited_from_model:
#             type = self.__init_type_by_loss__(loss)
#         print(">>>> Train %s..." % type)
#         self.__init_dataset__(type, emb_loss_names)
#         if self.train_ds is None:
#             print(">>>> [Error]: train_ds is None.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return
#         if self.is_distill_ds == False and type == self.distill:
#             print(">>>> [Error]: Dataset doesn't contain embedding data.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return

#         self.is_lr_on_batch = isinstance(self.lr_scheduler, myCallbacks.CosineLrScheduler)
#         if self.is_lr_on_batch:
#             self.lr_scheduler.steps_per_epoch = self.steps_per_epoch

#         basic_callbacks = [ii for ii in [self.my_history, self.model_checkpoint, self.lr_scheduler] if ii is not None]
#         self.callbacks = self.my_evals + self.custom_callbacks + basic_callbacks
#         # self.basic_model.trainable = True
#         self.__init_optimizer__(optimizer)
#         if not self.inited_from_model:
#             header_append_norm = isinstance(loss, losses.MagFaceLoss) or isinstance(loss, losses.AdaFaceLoss)
#             self.__init_model__(type, lossTopK, header_append_norm)

#         # loss_weights
#         self.cur_loss, self.loss_weights = [loss], {ii: lossWeight for ii in self.model.output_names}
#         if self.center in emb_loss_names and type != self.center:
#             loss_class = emb_loss_names[self.center]
#             print(">>>> Attach center loss:", loss_class.__name__)
#             emb_shape = self.basic_model.output_shape[-1]
#             initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
#             center_loss = loss_class(self.classes, emb_shape=emb_shape, initial_file=initial_file)
#             self.callbacks.append(center_loss.save_centers_callback)
#             self.__add_emb_output_to_model__(self.center, center_loss, emb_loss_weights[self.center])

#         if self.triplet in emb_loss_names and type != self.triplet:
#             loss_class = emb_loss_names[self.triplet]
#             print(">>>> Attach triplet loss: %s, alpha = %f..." % (loss_class.__name__, tripletAlpha))
#             triplet_loss = loss_class(alpha=tripletAlpha)
#             self.__add_emb_output_to_model__(self.triplet, triplet_loss, emb_loss_weights[self.triplet])

#         if self.is_distill_ds and type != self.distill:
#             distill_loss = emb_loss_names.get(self.distill, losses.distiller_loss_cosine)
#             print(">>>> Attach disill loss:", distill_loss.__name__)
#             self.__add_emb_output_to_model__(self.distill, distill_loss, emb_loss_weights.get(self.distill, 1))

#         print(">>>> loss_weights:", self.loss_weights)
#         self.metrics = {ii: None if "embedding" in ii else "accuracy" for ii in self.model.output_names}
#         # self.callbacks.append(keras.callbacks.TerminateOnNaN())
#         self.callbacks.append(myCallbacks.ExitOnNaN())  # Exit directly avoiding further saving
#         # self.callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs=None: keras.backend.clear_session()))
#         # self.callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs=None: self.basic_model.save("aa_epoch{}.h5".format(epoch))))

#         if self.vpl_start_iters > 0:  # VPL mode, needs the actual batch_size
#             loss.build(self.batch_size_per_replica)
#             self.callbacks.append(myCallbacks.VPLUpdateQueue())

#         if self.gently_stop:
#             self.callbacks.append(self.gently_stop)

#         if bottleneckOnly:
#             print(">>>> Train bottleneckOnly...")
#             self.basic_model.trainable = False
#             self.callbacks = self.callbacks[len(self.my_evals) :]  # Exclude evaluation callbacks
#             self.__basic_train__(epoch, initial_epoch=0)
#             self.basic_model.trainable = True
#         else:
#             self.__basic_train__(initial_epoch + epoch, initial_epoch=initial_epoch)

#         print(">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s" % (type, self.model.history.epoch, self.model.stop_training))
#         print(">>>> My history:")
#         self.my_history.print_hist()
#         latest_save_path = os.path.join("checkpoints", os.path.splitext(self.save_path)[0] + "_basic_model_latest.h5")
#         print(">>>> Saving latest basic model to:", latest_save_path)
#         self.basic_model.save(latest_save_path)

#     def train(self, train_schedule, initial_epoch=0):
#         train_schedule = [train_schedule] if isinstance(train_schedule, dict) else train_schedule
#         for sch in train_schedule:
#             for ii in ["centerloss", "triplet", "distill"]:
#                 if ii in sch:
#                     sch.setdefault("embLossTypes", []).append(ii)
#                     sch.setdefault("embLossWeights", []).append(sch.pop(ii))
#             if "alpha" in sch:
#                 sch["tripletAlpha"] = sch.pop("alpha")

#             self.train_single_scheduler(**sch, initial_epoch=initial_epoch)
#             initial_epoch += 0 if sch.get("bottleneckOnly", False) else sch["epoch"]

#             if self.model is None or self.model.stop_training == True:
#                 print(">>>> But it's an early stop, break...")
#                 break
#         return initial_epoch











# import sys
# sys.path.append("C:\\Users\\HP\\Documents\\zmine\\parttime\\ghostfacerepo\\projectrepo\\GhostFaceNets")

# import os
# print("Imported os")  # Debug print
# import data
# print("Imported data")  # Debug print
# import evals
# print("Imported evals")  # Debug print
# import losses
# print("Imported losses")  # Debug print
# import GhostFaceNets, GhostFaceNets_with_Bias
# print("Imported GhostFaceNets modules")  # Debug print
# import myCallbacks
# print("Imported myCallbacks")  # Debug print
# import tensorflow as tf
# print("Imported tensorflow")  # Debug print
# from tensorflow import keras
# print("Imported keras")  # Debug print
# import models
# print("Imported models")  # Debug print
# import argparse
# print("Imported argparse")  # Debug print

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# print("Starting script execution")  # Debug print

# class Train:
#     def __init__(
#         self,
#         data_path,
#         save_path,
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     ):
#         print("Entered Train.__init__")  # Debug print
#         from inspect import getmembers, isfunction, isclass

#         custom_objects.update(dict([ii for ii in getmembers(losses) if isfunction(ii[1]) or isclass(ii[1])]))
#         custom_objects.update({"NormDense": models.NormDense})
#         print("Custom objects updated")  # Debug print

#         self.model, self.basic_model, self.save_path, self.inited_from_model, self.sam_rho, self.pretrained = None, None, save_path, False, sam_rho, pretrained
#         self.vpl_start_iters, self.vpl_allowed_delta = vpl_start_iters, vpl_allowed_delta
#         print("Instance variables initialized")  # Debug print
        
#         # Initialize a new model with ghostnetv2 backbone
#         print("Initializing new model with ghostnetv2 backbone")
#         self.basic_model = GhostFaceNets.buildin_models(
#             stem_model="ghostnetv2",
#             input_shape=(112, 112, 3),
#             dropout=0,
#             emb_shape=512,
#             output_layer="GDC",
#             bn_momentum=0.99,
#             bn_epsilon=0.001,
#             weights=None  # No pre-trained weights
#         )
#         print("New model initialized as basic_model using ghostnetv2 backbone")

#         if self.basic_model is None:
#             print(
#                 "Initialize model by:\n"
#                 "| basic_model                                                     | model           |\n"
#                 "| --------------------------------------------------------------- | --------------- |\n"
#                 "| model structure                                                 | None            |\n"
#                 "| basic model .h5 file                                            | None            |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
#                 "| None                                                            | None            |\n"
#                 "* Both None for reload model from 'checkpoints/{}'\n".format(save_path)
#             )
#             return

#         # Losses
#         self.softmax, self.arcface, self.arcface_partial, self.triplet = "softmax", "arcface", "arcface_partial", "triplet"
#         self.center, self.distill = "center", "distill"
        
#         if output_weight_decay >= 1:
#             l2_weight_decay = 0
#             for ii in self.basic_model.layers:
#                 if hasattr(ii, "kernel_regularizer") and isinstance(ii.kernel_regularizer, keras.regularizers.L2):
#                     l2_weight_decay = ii.kernel_regularizer.l2
#                     break
#             print(">>>> L2 regularizer value from basic_model:", l2_weight_decay)
#             output_weight_decay *= l2_weight_decay * 2
#         self.output_weight_decay = output_weight_decay

#         self.batch_size, self.batch_size_per_replica = batch_size, batch_size
#         if tf.distribute.has_strategy():
#             strategy = tf.distribute.get_strategy()
#             self.batch_size = batch_size * strategy.num_replicas_in_sync
#             print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
#             self.data_options = tf.data.Options()
#             self.data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
#         # Evaluate
#         my_evals = [evals.eval_callback(self.basic_model, ii, batch_size=self.batch_size_per_replica, eval_freq=eval_freq) for ii in eval_paths]
#         if len(my_evals) != 0:
#             my_evals[-1].save_model = os.path.splitext(save_path)[0]
        
#         self.my_history, self.model_checkpoint, self.lr_scheduler, self.gently_stop = myCallbacks.basic_callbacks(
#             save_path,
#             my_evals,
#             lr=lr_base,
#             lr_decay=lr_decay,
#             lr_min=lr_min,
#             lr_decay_steps=lr_decay_steps,
#             lr_warmup_steps=lr_warmup_steps,
#         )
#         self.gently_stop = None  # may not working for windows
#         self.my_evals, self.custom_callbacks = my_evals, []
#         self.metrics = ["accuracy"]
#         self.default_optimizer = "adam"

#         self.data_path, self.random_status, self.image_per_class, self.mixup_alpha = data_path, random_status, image_per_class, mixup_alpha
#         self.random_cutout_mask_area, self.partial_fc_split, self.samples_per_mining = random_cutout_mask_area, partial_fc_split, samples_per_mining
#         self.train_ds, self.steps_per_epoch, self.classes, self.is_triplet_dataset = None, None, 0, False
#         self.teacher_model_interf, self.is_distill_ds = teacher_model_interf, False
#         self.distill_emb_map_layer = None
#         print("Initialization complete")  # Debug print

#     def __search_embedding_layer__(self, model):
#         for ii in range(1, 6):
#             if model.layers[-ii].name == "embedding":
#                 return -ii
    
#     def __init_dataset__(self, type, emb_loss_names):
#         init_as_triplet = self.triplet in emb_loss_names or type == self.triplet
#         is_offline_triplet = self.samples_per_mining > 0
#         if self.train_ds is not None and init_as_triplet == self.is_triplet_dataset and not self.is_distill_ds and not is_offline_triplet:
#             return

#         dataset_params = {
#             "data_path": self.data_path,
#             "batch_size": self.batch_size,
#             "random_status": self.random_status,
#             "random_cutout_mask_area": self.random_cutout_mask_area,
#             "image_per_class": self.image_per_class,
#             "mixup_alpha": self.mixup_alpha,
#             "teacher_model_interf": self.teacher_model_interf,
#         }
#         print(f"Initializing dataset with type: {type}, params: {dataset_params}")  # Debug print

#         if is_offline_triplet:
#             print(">>>> Init offline triplet dataset...")
#             aa = data.Triplet_dataset_offline(basic_model=self.basic_model, samples_per_mining=self.samples_per_mining, **dataset_params)
#             self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = False
#         elif init_as_triplet:
#             print(">>>> Init triplet dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 print(">>>> Combining tfrecord dataset with triplet is NOT recommended.")
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 aa = data.Triplet_dataset(**dataset_params)
#                 self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = True
#         else:
#             print(">>>> Init softmax dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 self.train_ds, self.steps_per_epoch = data.prepare_dataset(**dataset_params, partial_fc_split=self.partial_fc_split)
#             self.is_triplet_dataset = False
#         if self.train_ds is None:
#             print(">>>> [Error]: Dataset initialization failed, train_ds is None.")  # Debug print
#             return

#         if tf.distribute.has_strategy():
#             self.train_ds = self.train_ds.with_options(self.data_options)

#         label_spec = self.train_ds.element_spec[-1]
#         if isinstance(label_spec, tuple):
#             self.is_distill_ds = True
#             self.teacher_emb_size = label_spec[0].shape[-1]
#             self.classes = label_spec[1].shape[-1]
#             if type == self.distill:
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[1:] * len(emb_loss_names) + yy[:1]))
#             elif (self.distill in emb_loss_names and len(emb_loss_names) != 1) or (self.distill not in emb_loss_names and len(emb_loss_names) != 0):
#                 label_data_len = len(emb_loss_names) if self.distill in emb_loss_names else len(emb_loss_names) + 1
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[:1] + yy[1:] * label_data_len))
#         else:
#             self.is_distill_ds = False
#             self.classes = label_spec.shape[-1]
#         print(f"Dataset initialized: classes={self.classes}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
    
#     def __init_optimizer__(self, optimizer):
#         if optimizer == None:
#             if self.model != None and self.model.optimizer != None:
#                 self.optimizer = self.model.optimizer
#                 compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#                 print(">>>> Reuse optimizer from previous model:", compiled_opt.__class__.__name__)
#             else:
#                 print(">>>> Use default optimizer:", self.default_optimizer)
#                 self.optimizer = self.default_optimizer
#         else:
#             print(">>>> Use specified optimizer:", optimizer)
#             self.optimizer = optimizer

#         try:
#             import tensorflow_addons as tfa
#         except:
#             pass
#         else:
#             compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#             if isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
#                 print(">>>> Append weight decay callback...")
#                 lr_base, wd_base = self.optimizer.lr.numpy(), self.optimizer.weight_decay.numpy()
#                 wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=self.is_lr_on_batch)
#                 self.callbacks.append(wd_callback)

#     def __init_model__(self, type, loss_top_k=1, header_append_norm=False):
#         inputs = self.basic_model.inputs[0]
#         embedding = self.basic_model.outputs[0]
#         is_multi_output = lambda mm: len(mm.outputs) != 1 or isinstance(mm.layers[-1], keras.layers.Concatenate)
#         if self.model != None and is_multi_output(self.model):
#             output_layer = min(len(self.basic_model.layers), len(self.model.layers) - 1)
#             self.model = keras.models.Model(inputs, self.model.layers[output_layer].output)

#         if self.output_weight_decay != 0:
#             print(">>>> Add L2 regularizer to model output layer, output_weight_decay = %f" % self.output_weight_decay)
#             output_kernel_regularizer = keras.regularizers.L2(self.output_weight_decay / 2)
#         else:
#             output_kernel_regularizer = None

#         model_output_layer_name = None if self.model is None else self.model.output_names[-1]
#         if type == self.softmax and model_output_layer_name != self.softmax:
#             print(">>>> Add softmax layer...")
#             softmax_logits = keras.layers.Dense(self.classes, use_bias=False, name=self.softmax + "_logits", kernel_regularizer=output_kernel_regularizer)
#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 softmax_logits.build(embedding.shape)
#                 weight_cur = softmax_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     softmax_logits.set_weights(weight_pre)
#             logits = softmax_logits(embedding)
#             output_fp32 = keras.layers.Activation("softmax", dtype="float32", name=self.softmax)(logits)
#             self.model = keras.models.Model(inputs, output_fp32)
#         elif type == self.arcface and (model_output_layer_name != self.arcface or self.model.layers[-1].append_norm != header_append_norm):
#             vpl_start_iters = self.vpl_start_iters * self.steps_per_epoch if self.vpl_start_iters < 50 else self.vpl_start_iters
#             vpl_kwargs = {"vpl_lambda": 0.15, "start_iters": vpl_start_iters, "allowed_delta": self.vpl_allowed_delta}
#             arc_kwargs = {"loss_top_k": loss_top_k, "append_norm": header_append_norm, "partial_fc_split": self.partial_fc_split, "name": self.arcface}
#             print(">>>> Add arcface layer, arc_kwargs={}, vpl_kwargs={}...".format(arc_kwargs, vpl_kwargs))
#             if vpl_start_iters > 0:
#                 batch_size = self.batch_size_per_replica
#                 arcface_logits = models.NormDenseVPL(batch_size, self.classes, output_kernel_regularizer, **arc_kwargs, **vpl_kwargs, dtype="float32")
#             else:
#                 arcface_logits = models.NormDense(self.classes, output_kernel_regularizer, **arc_kwargs, dtype="float32")

#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 arcface_logits.build(embedding.shape)
#                 weight_cur = arcface_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     arcface_logits.set_weights(weight_pre)
#             output_fp32 = arcface_logits(embedding)
#             self.model = keras.models.Model(inputs, output_fp32)
#         elif type in [self.triplet, self.center, self.distill]:
#             self.model = self.basic_model
#             self.model.output_names[0] = type + "_embedding"
#         else:
#             print(">>>> Will NOT change model output layer.")

#         if self.pretrained is not None:
#             if self.model is None:
#                 self.basic_model.load_weights(self.pretrained)
#             else:
#                 self.model.load_weights(self.pretrained)
#             self.pretrained = None
#         print(f"Model initialization complete for type: {type}")  # Debug print

#     def __add_emb_output_to_model__(self, emb_type, emb_loss, emb_loss_weight):
#         nns = self.model.output_names
#         emb_shape = self.basic_model.output_shape[-1]
#         if emb_type == self.distill and self.teacher_emb_size != emb_shape:
#             print(">>>> Add a dense layer to map embedding: student %d --> teacher %d" % (emb_shape, self.teacher_emb_size))
#             embedding = self.basic_model.outputs[0]
#             if self.distill_emb_map_layer is None:
#                 self.distill_emb_map_layer = keras.layers.Dense(self.teacher_emb_size, use_bias=False, name="distill_map", dtype="float32")
#             emb_map_output = self.distill_emb_map_layer(embedding)
#             self.model = keras.models.Model(self.model.inputs[0], [emb_map_output] + self.model.outputs)
#         else:
#             self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)

#         self.model.output_names[0] = emb_type + "_embedding"
#         for id, nn in enumerate(nns):
#             self.model.output_names[id + 1] = nn
#         self.cur_loss = [emb_loss, *self.cur_loss]
#         self.loss_weights.update({self.model.output_names[0]: emb_loss_weight})

#     def __init_type_by_loss__(self, loss):
#         print(">>>> Init type by loss function name...")
#         if isinstance(loss, str):
#             return self.softmax

#         if loss.__class__.__name__ == "function":
#             ss = loss.__name__.lower()
#             if self.softmax in ss:
#                 return self.softmax
#             if self.arcface in ss:
#                 return self.arcface
#             if self.triplet in ss:
#                 return self.triplet
#             if self.distill in ss:
#                 return self.distill
#         else:
#             ss = loss.__class__.__name__.lower()
#             if isinstance(loss, losses.TripletLossWapper) or self.triplet in ss:
#                 return self.triplet
#             if isinstance(loss, losses.CenterLoss) or self.center in ss:
#                 return self.center
#             if isinstance(loss, losses.ArcfaceLoss) or self.arcface in ss:
#                 return self.arcface
#             if isinstance(loss, losses.ArcfaceLossSimple) or isinstance(loss, losses.AdaCosLoss):
#                 return self.arcface
#             if isinstance(loss, losses.DistillKLDivergenceLoss):
#                 return self.arcface
#             if self.softmax in ss:
#                 return self.softmax
#         return self.softmax

#     def __init_emb_losses__(self, embLossTypes=None, embLossWeights=1):
#         emb_loss_names, emb_loss_weights = {}, {}
#         if embLossTypes is not None:
#             embLossTypes = embLossTypes if isinstance(embLossTypes, list) else [embLossTypes]
#             for id, ee in enumerate(embLossTypes):
#                 emb_loss_name = ee.lower() if isinstance(ee, str) else ee.__name__.lower()
#                 emb_loss_weight = float(embLossWeights[id] if isinstance(embLossWeights, list) else embLossWeights)
#                 if "centerloss" in emb_loss_name:
#                     emb_loss_names[self.center] = losses.CenterLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.center] = emb_loss_weight
#                 elif "triplet" in emb_loss_name:
#                     emb_loss_names[self.triplet] = losses.BatchHardTripletLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.triplet] = emb_loss_weight
#                 elif "distill" in emb_loss_name:
#                     emb_loss_names[self.distill] = losses.distiller_loss_cosine if ee == None or isinstance(ee, str) else ee
#                     emb_loss_weights[self.distill] = emb_loss_weight
#         return emb_loss_names, emb_loss_weights

#     def __basic_train__(self, epochs, initial_epoch=0):
#         self.model.compile(optimizer=self.optimizer, loss=self.cur_loss, metrics=self.metrics, loss_weights=self.loss_weights)
#         print(f"Starting model training: epochs={epochs}, initial_epoch={initial_epoch}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
#         self.model.fit(
#             self.train_ds,
#             epochs=epochs,
#             verbose=1,
#             callbacks=self.callbacks,
#             initial_epoch=initial_epoch,
#             steps_per_epoch=self.steps_per_epoch,
#             use_multiprocessing=True,
#             workers=4,
#         )
#         print("Training completed")  # Debug print

#     def reset_dataset(self, data_path=None):
#         self.train_ds = None
#         if data_path != None:
#             self.data_path = data_path

#     def train_single_scheduler(
#         self,
#         epoch,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35,
#     ):
#         emb_loss_names, emb_loss_weights = self.__init_emb_losses__(embLossTypes, embLossWeights)

#         # If no loss is provided, use a default ArcFace loss
#         if loss is None:
#             print(">>>> No loss specified, using default ArcFace loss...")
#             loss = losses.ArcfaceLoss()
#             if self.model is not None and self.model.built:
#                 print(">>>> Compiling model with default ArcFace loss...")
#                 self.model.compile(
#                     optimizer="adam",
#                     loss=loss,
#                     metrics=["accuracy"]
#                 )

#         if type is None and not self.inited_from_model:
#             type = self.__init_type_by_loss__(loss)
#         print(">>>> Train %s..." % type)
#         self.__init_dataset__(type, emb_loss_names)
#         if self.train_ds is None:
#             print(">>>> [Error]: train_ds is None.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return
#         if self.is_distill_ds == False and type == self.distill:
#             print(">>>> [Error]: Dataset doesn't contain embedding data.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return

#         self.is_lr_on_batch = isinstance(self.lr_scheduler, myCallbacks.CosineLrScheduler)
#         if self.is_lr_on_batch:
#             self.lr_scheduler.steps_per_epoch = self.steps_per_epoch

#         basic_callbacks = [ii for ii in [self.my_history, self.model_checkpoint, self.lr_scheduler] if ii is not None]
#         self.callbacks = self.my_evals + self.custom_callbacks + basic_callbacks
#         self.__init_optimizer__(optimizer)
#         if not self.inited_from_model:
#             header_append_norm = isinstance(loss, losses.MagFaceLoss) or isinstance(loss, losses.AdaFaceLoss)
#             self.__init_model__(type, lossTopK, header_append_norm)

#         self.cur_loss, self.loss_weights = [loss], {ii: lossWeight for ii in self.model.output_names}
#         if self.center in emb_loss_names and type != self.center:
#             loss_class = emb_loss_names[self.center]
#             print(">>>> Attach center loss:", loss_class.__name__)
#             emb_shape = self.basic_model.output_shape[-1]
#             initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
#             center_loss = loss_class(self.classes, emb_shape=emb_shape, initial_file=initial_file)
#             self.callbacks.append(center_loss.save_centers_callback)
#             self.__add_emb_output_to_model__(self.center, center_loss, emb_loss_weights[self.center])

#         if self.triplet in emb_loss_names and type != self.triplet:
#             loss_class = emb_loss_names[self.triplet]
#             print(">>>> Attach triplet loss: %s, alpha = %f..." % (loss_class.__name__, tripletAlpha))
#             triplet_loss = loss_class(alpha=tripletAlpha)
#             self.__add_emb_output_to_model__(self.triplet, triplet_loss, emb_loss_weights[self.triplet])

#         if self.is_distill_ds and type != self.distill:
#             distill_loss = emb_loss_names.get(self.distill, losses.distiller_loss_cosine)
#             print(">>>> Attach distill loss:", distill_loss.__name__)
#             self.__add_emb_output_to_model__(self.distill, distill_loss, emb_loss_weights.get(self.distill, 1))

#         print(">>>> loss_weights:", self.loss_weights)
#         self.metrics = {ii: None if "embedding" in ii else "accuracy" for ii in self.model.output_names}
#         self.callbacks.append(myCallbacks.ExitOnNaN())
#         if self.vpl_start_iters > 0:
#             loss.build(self.batch_size_per_replica)
#             self.callbacks.append(myCallbacks.VPLUpdateQueue())

#         if self.gently_stop:
#             self.callbacks.append(self.gently_stop)

#         if bottleneckOnly:
#             print(">>>> Train bottleneckOnly...")
#             self.basic_model.trainable = False
#             self.callbacks = self.callbacks[len(self.my_evals) :]
#             self.__basic_train__(epoch, initial_epoch=0)
#             self.basic_model.trainable = True
#         else:
#             self.__basic_train__(initial_epoch + epoch, initial_epoch=initial_epoch)

#         print(">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s" % (type, self.model.history.epoch, self.model.stop_training))
#         print(">>>> My history:")
#         self.my_history.print_hist()
#         latest_save_path = os.path.join("checkpoints", os.path.splitext(self.save_path)[0] + "_basic_model_latest.h5")
#         print(">>>> Saving latest basic model to:", latest_save_path)
#         self.basic_model.save(latest_save_path)

#     def train(self, train_schedule, initial_epoch=0):
#         train_schedule = [train_schedule] if isinstance(train_schedule, dict) else train_schedule
#         for sch in train_schedule:
#             for ii in ["centerloss", "triplet", "distill"]:
#                 if ii in sch:
#                     sch.setdefault("embLossTypes", []).append(ii)
#                     sch.setdefault("embLossWeights", []).append(sch.pop(ii))
#             if "alpha" in sch:
#                 sch["tripletAlpha"] = sch.pop("alpha")

#             self.train_single_scheduler(**sch, initial_epoch=initial_epoch)
#             initial_epoch += 0 if sch.get("bottleneckOnly", False) else sch["epoch"]

#             if self.model is None or self.model.stop_training == True:
#                 print(">>>> But it's an early stop, break...")
#                 break
#         return initial_epoch

# if __name__ == "__main__":
#     print("Entering main execution block")  # Debug print
#     parser = argparse.ArgumentParser(description="Train GhostFaceNets model")
#     parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
#     parser.add_argument('--model', type=str, default=None, help='Path to model .h5 file or model name (ignored, training from scratch)')
#     args = parser.parse_args()
#     print(f"Parsed arguments: data_dir={args.data_dir}, model={args.model}")  # Debug print
#     trainer = Train(
#         data_path=args.data_dir,
#         save_path="train",
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     )
#     print("Train instance created")  # Debug print
#     trainer.train_single_scheduler(
#         epoch=10,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35
#     )




























# import sys
# sys.path.append("C:\\Users\\HP\\Documents\\zmine\\parttime\\ghostfacerepo\\projectrepo\\GhostFaceNets")

# import os
# print("Imported os")  # Debug print
# import data
# print("Imported data")  # Debug print
# import evals
# print("Imported evals")  # Debug print
# import losses
# print("Imported losses")  # Debug print
# import GhostFaceNets, GhostFaceNets_with_Bias
# print("Imported GhostFaceNets modules")  # Debug print
# import myCallbacks
# print("Imported myCallbacks")  # Debug print
# import tensorflow as tf
# print("Imported tensorflow")  # Debug print
# from tensorflow import keras
# print("Imported keras")  # Debug print
# import models
# print("Imported models")  # Debug print
# import argparse
# print("Imported argparse")  # Debug print

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# print("Starting script execution")  # Debug print

# class Train:
#     def __init__(
#         self,
#         data_path,
#         save_path,
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     ):
#         print("Entered Train.__init__")  # Debug print
#         from inspect import getmembers, isfunction, isclass

#         custom_objects.update(dict([ii for ii in getmembers(losses) if isfunction(ii[1]) or isclass(ii[1])]))
#         custom_objects.update({"NormDense": models.NormDense})
#         print("Custom objects updated")  # Debug print

#         self.model, self.basic_model, self.save_path, self.inited_from_model, self.sam_rho, self.pretrained = None, None, save_path, False, sam_rho, pretrained
#         self.vpl_start_iters, self.vpl_allowed_delta = vpl_start_iters, vpl_allowed_delta
#         print("Instance variables initialized")  # Debug print
        
#         # Initialize a new model with ghostnetv2 backbone
#         print("Initializing new model with ghostnetv2 backbone")
#         self.basic_model = GhostFaceNets.buildin_models(
#             stem_model="ghostnetv2",
#             input_shape=(112, 112, 3),
#             dropout=0,
#             emb_shape=512,
#             output_layer="GDC",
#             bn_momentum=0.99,
#             bn_epsilon=0.001,
#             weights=None  # No pre-trained weights
#         )
#         print("New model initialized as basic_model using ghostnetv2 backbone")
#         # Print the summary of the basic model (includes FocalModulationBlock)
#         print(">>>> Printing basic_model summary (includes FocalModulationBlock):")
#         self.basic_model.summary()

#         if self.basic_model is None:
#             print(
#                 "Initialize model by:\n"
#                 "| basic_model                                                     | model           |\n"
#                 "| --------------------------------------------------------------- | --------------- |\n"
#                 "| model structure                                                 | None            |\n"
#                 "| basic model .h5 file                                            | None            |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
#                 "| None                                                            | None            |\n"
#                 "* Both None for reload model from 'checkpoints/{}'\n".format(save_path)
#             )
#             return

#         # Losses
#         self.softmax, self.arcface, self.arcface_partial, self.triplet = "softmax", "arcface", "arcface_partial", "triplet"
#         self.center, self.distill = "center", "distill"
        
#         if output_weight_decay >= 1:
#             l2_weight_decay = 0
#             for ii in self.basic_model.layers:
#                 if hasattr(ii, "kernel_regularizer") and isinstance(ii.kernel_regularizer, keras.regularizers.L2):
#                     l2_weight_decay = ii.kernel_regularizer.l2
#                     break
#             print(">>>> L2 regularizer value from basic_model:", l2_weight_decay)
#             output_weight_decay *= l2_weight_decay * 2
#         self.output_weight_decay = output_weight_decay

#         self.batch_size, self.batch_size_per_replica = batch_size, batch_size
#         if tf.distribute.has_strategy():
#             strategy = tf.distribute.get_strategy()
#             self.batch_size = batch_size * strategy.num_replicas_in_sync
#             print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
#             self.data_options = tf.data.Options()
#             self.data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
#         # Evaluate
#         my_evals = [evals.eval_callback(self.basic_model, ii, batch_size=self.batch_size_per_replica, eval_freq=eval_freq) for ii in eval_paths]
#         if len(my_evals) != 0:
#             my_evals[-1].save_model = os.path.splitext(save_path)[0]
        
#         self.my_history, self.model_checkpoint, self.lr_scheduler, self.gently_stop = myCallbacks.basic_callbacks(
#             save_path,
#             my_evals,
#             lr=lr_base,
#             lr_decay=lr_decay,
#             lr_min=lr_min,
#             lr_decay_steps=lr_decay_steps,
#             lr_warmup_steps=lr_warmup_steps,
#         )
#         self.gently_stop = None  # may not working for windows
#         self.my_evals, self.custom_callbacks = my_evals, []
#         self.metrics = ["accuracy"]
#         self.default_optimizer = "adam"

#         self.data_path, self.random_status, self.image_per_class, self.mixup_alpha = data_path, random_status, image_per_class, mixup_alpha
#         self.random_cutout_mask_area, self.partial_fc_split, self.samples_per_mining = random_cutout_mask_area, partial_fc_split, samples_per_mining
#         self.train_ds, self.steps_per_epoch, self.classes, self.is_triplet_dataset = None, None, 0, False
#         self.teacher_model_interf, self.is_distill_ds = teacher_model_interf, False
#         self.distill_emb_map_layer = None
#         print("Initialization complete")  # Debug print

#     def __search_embedding_layer__(self, model):
#         for ii in range(1, 6):
#             if model.layers[-ii].name == "embedding":
#                 return -ii
    
#     def __init_dataset__(self, type, emb_loss_names):
#         init_as_triplet = self.triplet in emb_loss_names or type == self.triplet
#         is_offline_triplet = self.samples_per_mining > 0
#         if self.train_ds is not None and init_as_triplet == self.is_triplet_dataset and not self.is_distill_ds and not is_offline_triplet:
#             return

#         dataset_params = {
#             "data_path": self.data_path,
#             "batch_size": self.batch_size,
#             "random_status": self.random_status,
#             "random_cutout_mask_area": self.random_cutout_mask_area,
#             "image_per_class": self.image_per_class,
#             "mixup_alpha": self.mixup_alpha,
#             "teacher_model_interf": self.teacher_model_interf,
#         }
#         print(f"Initializing dataset with type: {type}, params: {dataset_params}")  # Debug print

#         if is_offline_triplet:
#             print(">>>> Init offline triplet dataset...")
#             aa = data.Triplet_dataset_offline(basic_model=self.basic_model, samples_per_mining=self.samples_per_mining, **dataset_params)
#             self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = False
#         elif init_as_triplet:
#             print(">>>> Init triplet dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 print(">>>> Combining tfrecord dataset with triplet is NOT recommended.")
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 aa = data.Triplet_dataset(**dataset_params)
#                 self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = True
#         else:
#             print(">>>> Init softmax dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 self.train_ds, self.steps_per_epoch = data.prepare_dataset(**dataset_params, partial_fc_split=self.partial_fc_split)
#             self.is_triplet_dataset = False
#         if self.train_ds is None:
#             print(">>>> [Error]: Dataset initialization failed, train_ds is None.")  # Debug print
#             return

#         if tf.distribute.has_strategy():
#             self.train_ds = self.train_ds.with_options(self.data_options)

#         label_spec = self.train_ds.element_spec[-1]
#         if isinstance(label_spec, tuple):
#             self.is_distill_ds = True
#             self.teacher_emb_size = label_spec[0].shape[-1]
#             self.classes = label_spec[1].shape[-1]
#             if type == self.distill:
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[1:] * len(emb_loss_names) + yy[:1]))
#             elif (self.distill in emb_loss_names and len(emb_loss_names) != 1) or (self.distill not in emb_loss_names and len(emb_loss_names) != 0):
#                 label_data_len = len(emb_loss_names) if self.distill in emb_loss_names else len(emb_loss_names) + 1
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[:1] + yy[1:] * label_data_len))
#         else:
#             self.is_distill_ds = False
#             self.classes = label_spec.shape[-1]
#         print(f"Dataset initialized: classes={self.classes}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
    
#     def __init_optimizer__(self, optimizer):
#         if optimizer == None:
#             if self.model != None and self.model.optimizer != None:
#                 self.optimizer = self.model.optimizer
#                 compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#                 print(">>>> Reuse optimizer from previous model:", compiled_opt.__class__.__name__)
#             else:
#                 print(">>>> Use default optimizer:", self.default_optimizer)
#                 self.optimizer = self.default_optimizer
#         else:
#             print(">>>> Use specified optimizer:", optimizer)
#             self.optimizer = optimizer

#         try:
#             import tensorflow_addons as tfa
#         except:
#             pass
#         else:
#             compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#             if isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
#                 print(">>>> Append weight decay callback...")
#                 lr_base, wd_base = self.optimizer.lr.numpy(), self.optimizer.weight_decay.numpy()
#                 wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=self.is_lr_on_batch)
#                 self.callbacks.append(wd_callback)

#     def __init_model__(self, type, loss_top_k=1, header_append_norm=False):
#         inputs = self.basic_model.inputs[0]
#         embedding = self.basic_model.outputs[0]
#         is_multi_output = lambda mm: len(mm.outputs) != 1 or isinstance(mm.layers[-1], keras.layers.Concatenate)
#         if self.model != None and is_multi_output(self.model):
#             output_layer = min(len(self.basic_model.layers), len(self.model.layers) - 1)
#             self.model = keras.models.Model(inputs, self.model.layers[output_layer].output)

#         if self.output_weight_decay != 0:
#             print(">>>> Add L2 regularizer to model output layer, output_weight_decay = %f" % self.output_weight_decay)
#             output_kernel_regularizer = keras.regularizers.L2(self.output_weight_decay / 2)
#         else:
#             output_kernel_regularizer = None

#         model_output_layer_name = None if self.model is None else self.model.output_names[-1]
#         if type == self.softmax and model_output_layer_name != self.softmax:
#             print(">>>> Add softmax layer...")
#             softmax_logits = keras.layers.Dense(self.classes, use_bias=False, name=self.softmax + "_logits", kernel_regularizer=output_kernel_regularizer)
#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 softmax_logits.build(embedding.shape)
#                 weight_cur = softmax_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     softmax_logits.set_weights(weight_pre)
#             logits = softmax_logits(embedding)
#             output_fp32 = keras.layers.Activation("softmax", dtype="float32", name=self.softmax)(logits)
#             self.model = keras.models.Model(inputs, output_fp32)
#         elif type == self.arcface and (model_output_layer_name != self.arcface or self.model.layers[-1].append_norm != header_append_norm):
#             vpl_start_iters = self.vpl_start_iters * self.steps_per_epoch if self.vpl_start_iters < 50 else self.vpl_start_iters
#             vpl_kwargs = {"vpl_lambda": 0.15, "start_iters": vpl_start_iters, "allowed_delta": self.vpl_allowed_delta}
#             arc_kwargs = {"loss_top_k": loss_top_k, "append_norm": header_append_norm, "partial_fc_split": self.partial_fc_split, "name": self.arcface}
#             print(">>>> Add arcface layer, arc_kwargs={}, vpl_kwargs={}...".format(arc_kwargs, vpl_kwargs))
#             if vpl_start_iters > 0:
#                 batch_size = self.batch_size_per_replica
#                 arcface_logits = models.NormDenseVPL(batch_size, self.classes, output_kernel_regularizer, **arc_kwargs, **vpl_kwargs, dtype="float32")
#             else:
#                 arcface_logits = models.NormDense(self.classes, output_kernel_regularizer, **arc_kwargs, dtype="float32")

#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 arcface_logits.build(embedding.shape)
#                 weight_cur = arcface_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     arcface_logits.set_weights(weight_pre)
#             output_fp32 = arcface_logits(embedding)
#             self.model = keras.models.Model(inputs, output_fp32)
#             # Print the summary of the full model (with ArcFace layer)
#             print(">>>> Printing full model summary (with ArcFace layer):")
#             self.model.summary()
#         elif type in [self.triplet, self.center, self.distill]:
#             self.model = self.basic_model
#             self.model.output_names[0] = type + "_embedding"
#         else:
#             print(">>>> Will NOT change model output layer.")

#         if self.pretrained is not None:
#             if self.model is None:
#                 self.basic_model.load_weights(self.pretrained)
#             else:
#                 self.model.load_weights(self.pretrained)
#             self.pretrained = None
#         print(f"Model initialization complete for type: {type}")  # Debug print

#     def __add_emb_output_to_model__(self, emb_type, emb_loss, emb_loss_weight):
#         nns = self.model.output_names
#         emb_shape = self.basic_model.output_shape[-1]
#         if emb_type == self.distill and self.teacher_emb_size != emb_shape:
#             print(">>>> Add a dense layer to map embedding: student %d --> teacher %d" % (emb_shape, self.teacher_emb_size))
#             embedding = self.basic_model.outputs[0]
#             if self.distill_emb_map_layer is None:
#                 self.distill_emb_map_layer = keras.layers.Dense(self.teacher_emb_size, use_bias=False, name="distill_map", dtype="float32")
#             emb_map_output = self.distill_emb_map_layer(embedding)
#             self.model = keras.models.Model(self.model.inputs[0], [emb_map_output] + self.model.outputs)
#         else:
#             self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)

#         self.model.output_names[0] = emb_type + "_embedding"
#         for id, nn in enumerate(nns):
#             self.model.output_names[id + 1] = nn
#         self.cur_loss, self.loss_weights = [emb_loss, *self.cur_loss], {ii: lossWeight for ii in self.model.output_names}
#         self.loss_weights.update({self.model.output_names[0]: emb_loss_weight})

#     def __init_type_by_loss__(self, loss):
#         print(">>>> Init type by loss function name...")
#         if isinstance(loss, str):
#             return self.softmax

#         if loss.__class__.__name__ == "function":
#             ss = loss.__name__.lower()
#             if self.softmax in ss:
#                 return self.softmax
#             if self.arcface in ss:
#                 return self.arcface
#             if self.triplet in ss:
#                 return self.triplet
#             if self.distill in ss:
#                 return self.distill
#         else:
#             ss = loss.__class__.__name__.lower()
#             if isinstance(loss, losses.TripletLossWapper) or self.triplet in ss:
#                 return self.triplet
#             if isinstance(loss, losses.CenterLoss) or self.center in ss:
#                 return self.center
#             if isinstance(loss, losses.ArcfaceLoss) or self.arcface in ss:
#                 return self.arcface
#             if isinstance(loss, losses.ArcfaceLossSimple) or isinstance(loss, losses.AdaCosLoss):
#                 return self.arcface
#             if isinstance(loss, losses.DistillKLDivergenceLoss):
#                 return self.arcface
#             if self.softmax in ss:
#                 return self.softmax
#         return self.softmax

#     def __init_emb_losses__(self, embLossTypes=None, embLossWeights=1):
#         emb_loss_names, emb_loss_weights = {}, {}
#         if embLossTypes is not None:
#             embLossTypes = embLossTypes if isinstance(embLossTypes, list) else [embLossTypes]
#             for id, ee in enumerate(embLossTypes):
#                 emb_loss_name = ee.lower() if isinstance(ee, str) else ee.__name__.lower()
#                 emb_loss_weight = float(embLossWeights[id] if isinstance(embLossWeights, list) else embLossWeights)
#                 if "centerloss" in emb_loss_name:
#                     emb_loss_names[self.center] = losses.CenterLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.center] = emb_loss_weight
#                 elif "triplet" in emb_loss_name:
#                     emb_loss_names[self.triplet] = losses.BatchHardTripletLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.triplet] = emb_loss_weight
#                 elif "distill" in emb_loss_name:
#                     emb_loss_names[self.distill] = losses.distiller_loss_cosine if ee == None or isinstance(ee, str) else ee
#                     emb_loss_weights[self.distill] = emb_loss_weight
#         return emb_loss_names, emb_loss_weights

#     def __basic_train__(self, epochs, initial_epoch=0):
#         self.model.compile(optimizer=self.optimizer, loss=self.cur_loss, metrics=self.metrics, loss_weights=self.loss_weights)
#         print(f"Starting model training: epochs={epochs}, initial_epoch={initial_epoch}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
#         self.model.fit(
#             self.train_ds,
#             epochs=epochs,
#             verbose=1,
#             callbacks=self.callbacks,
#             initial_epoch=initial_epoch,
#             steps_per_epoch=self.steps_per_epoch,
#             use_multiprocessing=True,
#             workers=4,
#         )
#         print("Training completed")  # Debug print

#     def reset_dataset(self, data_path=None):
#         self.train_ds = None
#         if data_path != None:
#             self.data_path = data_path

#     def train_single_scheduler(
#         self,
#         epoch,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35,
#     ):
#         emb_loss_names, emb_loss_weights = self.__init_emb_losses__(embLossTypes, embLossWeights)

#         # If no loss is provided, use a default ArcFace loss
#         if loss is None:
#             print(">>>> No loss specified, using default ArcFace loss...")
#             loss = losses.ArcfaceLoss()
#             if self.model is not None and self.model.built:
#                 print(">>>> Compiling model with default ArcFace loss...")
#                 self.model.compile(
#                     optimizer="adam",
#                     loss=loss,
#                     metrics=["accuracy"]
#                 )

#         if type is None and not self.inited_from_model:
#             type = self.__init_type_by_loss__(loss)
#         print(">>>> Train %s..." % type)
#         self.__init_dataset__(type, emb_loss_names)
#         if self.train_ds is None:
#             print(">>>> [Error]: train_ds is None.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return
#         if self.is_distill_ds == False and type == self.distill:
#             print(">>>> [Error]: Dataset doesn't contain embedding data.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return

#         self.is_lr_on_batch = isinstance(self.lr_scheduler, myCallbacks.CosineLrScheduler)
#         if self.is_lr_on_batch:
#             self.lr_scheduler.steps_per_epoch = self.steps_per_epoch

#         basic_callbacks = [ii for ii in [self.my_history, self.model_checkpoint, self.lr_scheduler] if ii is not None]
#         self.callbacks = self.my_evals + self.custom_callbacks + basic_callbacks
#         self.__init_optimizer__(optimizer)
#         if not self.inited_from_model:
#             header_append_norm = isinstance(loss, losses.MagFaceLoss) or isinstance(loss, losses.AdaFaceLoss)
#             self.__init_model__(type, lossTopK, header_append_norm)

#         self.cur_loss, self.loss_weights = [loss], {ii: lossWeight for ii in self.model.output_names}
#         if self.center in emb_loss_names and type != self.center:
#             loss_class = emb_loss_names[self.center]
#             print(">>>> Attach center loss:", loss_class.__name__)
#             emb_shape = self.basic_model.output_shape[-1]
#             initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
#             center_loss = loss_class(self.classes, emb_shape=emb_shape, initial_file=initial_file)
#             self.callbacks.append(center_loss.save_centers_callback)
#             self.__add_emb_output_to_model__(self.center, center_loss, emb_loss_weights[self.center])

#         if self.triplet in emb_loss_names and type != self.triplet:
#             loss_class = emb_loss_names[self.triplet]
#             print(">>>> Attach triplet loss: %s, alpha = %f..." % (loss_class.__name__, tripletAlpha))
#             triplet_loss = loss_class(alpha=tripletAlpha)
#             self.__add_emb_output_to_model__(self.triplet, triplet_loss, emb_loss_weights[self.triplet])

#         if self.is_distill_ds and type != self.distill:
#             distill_loss = emb_loss_names.get(self.distill, losses.distiller_loss_cosine)
#             print(">>>> Attach distill loss:", distill_loss.__name__)
#             self.__add_emb_output_to_model__(self.distill, distill_loss, emb_loss_weights.get(self.distill, 1))

#         print(">>>> loss_weights:", self.loss_weights)
#         self.metrics = {ii: None if "embedding" in ii else "accuracy" for ii in self.model.output_names}
#         self.callbacks.append(myCallbacks.ExitOnNaN())
#         if self.vpl_start_iters > 0:
#             loss.build(self.batch_size_per_replica)
#             self.callbacks.append(myCallbacks.VPLUpdateQueue())

#         if self.gently_stop:
#             self.callbacks.append(self.gently_stop)

#         if bottleneckOnly:
#             print(">>>> Train bottleneckOnly...")
#             self.basic_model.trainable = False
#             self.callbacks = self.callbacks[len(self.my_evals) :]
#             self.__basic_train__(epoch, initial_epoch=0)
#             self.basic_model.trainable = True
#         else:
#             self.__basic_train__(initial_epoch + epoch, initial_epoch=initial_epoch)

#         print(">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s" % (type, self.model.history.epoch, self.model.stop_training))
#         print(">>>> My history:")
#         self.my_history.print_hist()
#         latest_save_path = os.path.join("checkpoints", os.path.splitext(self.save_path)[0] + "_basic_model_latest.h5")
#         print(">>>> Saving latest basic model to:", latest_save_path)
#         self.basic_model.save(latest_save_path)

#     def train(self, train_schedule, initial_epoch=0):
#         train_schedule = [train_schedule] if isinstance(train_schedule, dict) else train_schedule
#         for sch in train_schedule:
#             for ii in ["centerloss", "triplet", "distill"]:
#                 if ii in sch:
#                     sch.setdefault("embLossTypes", []).append(ii)
#                     sch.setdefault("embLossWeights", []).append(sch.pop(ii))
#             if "alpha" in sch:
#                 sch["tripletAlpha"] = sch.pop("alpha")

#             self.train_single_scheduler(**sch, initial_epoch=initial_epoch)
#             initial_epoch += 0 if sch.get("bottleneckOnly", False) else sch["epoch"]

#             if self.model is None or self.model.stop_training == True:
#                 print(">>>> But it's an early stop, break...")
#                 break
#         return initial_epoch

# if __name__ == "__main__":
#     print("Entering main execution block")  # Debug print
#     parser = argparse.ArgumentParser(description="Train GhostFaceNets model")
#     parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
#     parser.add_argument('--model', type=str, default=None, help='Path to model .h5 file or model name (ignored, training from scratch)')
#     args = parser.parse_args()
#     print(f"Parsed arguments: data_dir={args.data_dir}, model={args.model}")  # Debug print
#     trainer = Train(
#         data_path=args.data_dir,
#         save_path="train",
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     )
#     print("Train instance created")  # Debug print
#     trainer.train_single_scheduler(
#         epoch=10,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35
#     )










#takes datapath as input from user 

# import sys
# sys.path.append("C:\\Users\\HP\\Documents\\zmine\\parttime\\ghostfacerepo\\projectrepo\\GhostFaceNets")

# import os
# print("Imported os")  # Debug print
# import data
# print("Imported data")  # Debug print
# import evals
# print("Imported evals")  # Debug print
# import losses
# print("Imported losses")  # Debug print
# import GhostFaceNets, GhostFaceNets_with_Bias
# print("Imported GhostFaceNets modules")  # Debug print
# import myCallbacks
# print("Imported myCallbacks")  # Debug print
# import tensorflow as tf
# print("Imported tensorflow")  # Debug print
# from tensorflow import keras
# print("Imported keras")  # Debug print
# import models
# print("Imported models")  # Debug print
# import argparse
# print("Imported argparse")  # Debug print

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# print("Starting script execution")  # Debug print

# class Train:
#     def __init__(
#         self,
#         data_path,
#         save_path,
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     ):
#         print("Entered Train.__init__")  # Debug print
#         from inspect import getmembers, isfunction, isclass

#         custom_objects.update(dict([ii for ii in getmembers(losses) if isfunction(ii[1]) or isclass(ii[1])]))
#         custom_objects.update({"NormDense": models.NormDense})
#         print("Custom objects updated")  # Debug print

#         self.model, self.basic_model, self.save_path, self.inited_from_model, self.sam_rho, self.pretrained = None, None, save_path, False, sam_rho, pretrained
#         self.vpl_start_iters, self.vpl_allowed_delta = vpl_start_iters, vpl_allowed_delta
#         print("Instance variables initialized")  # Debug print
        
#         # Initialize a new model with ghostnetv2 backbone
#         print("Initializing new model with ghostnetv2 backbone")
#         self.basic_model = GhostFaceNets.buildin_models(
#             stem_model="ghostnetv2",
#             input_shape=(112, 112, 3),
#             dropout=0,
#             emb_shape=512,
#             output_layer="GDC",
#             bn_momentum=0.99,
#             bn_epsilon=0.001,
#             weights=None  # No pre-trained weights
#         )
#         print("New model initialized as basic_model using ghostnetv2 backbone")
#         # Print the summary of the basic model (includes FocalModulationBlock)
#         print(">>>> Printing basic_model summary (includes FocalModulationBlock):")
#         self.basic_model.summary()

#         if self.basic_model is None:
#             print(
#                 "Initialize model by:\n"
#                 "| basic_model                                                     | model           |\n"
#                 "| --------------------------------------------------------------- | --------------- |\n"
#                 "| model structure                                                 | None            |\n"
#                 "| basic model .h5 file                                            | None            |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
#                 "| None                                                            | None            |\n"
#                 "* Both None for reload model from 'checkpoints/{}'\n".format(save_path)
#             )
#             return

#         # Losses
#         self.softmax, self.arcface, self.arcface_partial, self.triplet = "softmax", "arcface", "arcface_partial", "triplet"
#         self.center, self.distill = "center", "distill"
        
#         if output_weight_decay >= 1:
#             l2_weight_decay = 0
#             for ii in self.basic_model.layers:
#                 if hasattr(ii, "kernel_regularizer") and isinstance(ii.kernel_regularizer, keras.regularizers.L2):
#                     l2_weight_decay = ii.kernel_regularizer.l2
#                     break
#             print(">>>> L2 regularizer value from basic_model:", l2_weight_decay)
#             output_weight_decay *= l2_weight_decay * 2
#         self.output_weight_decay = output_weight_decay

#         self.batch_size, self.batch_size_per_replica = batch_size, batch_size
#         if tf.distribute.has_strategy():
#             strategy = tf.distribute.get_strategy()
#             self.batch_size = batch_size * strategy.num_replicas_in_sync
#             print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
#             self.data_options = tf.data.Options()
#             self.data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
#         # Evaluate
#         my_evals = [evals.eval_callback(self.basic_model, ii, batch_size=self.batch_size_per_replica, eval_freq=eval_freq) for ii in eval_paths]
#         if len(my_evals) != 0:
#             my_evals[-1].save_model = os.path.splitext(save_path)[0]
        
#         self.my_history, self.model_checkpoint, self.lr_scheduler, self.gently_stop = myCallbacks.basic_callbacks(
#             save_path,
#             my_evals,
#             lr=lr_base,
#             lr_decay=lr_decay,
#             lr_min=lr_min,
#             lr_decay_steps=lr_decay_steps,
#             lr_warmup_steps=lr_warmup_steps,
#         )
#         self.gently_stop = None  # may not working for windows
#         self.my_evals, self.custom_callbacks = my_evals, []
#         self.metrics = ["accuracy"]
#         self.default_optimizer = "adam"

#         self.data_path, self.random_status, self.image_per_class, self.mixup_alpha = data_path, random_status, image_per_class, mixup_alpha
#         self.random_cutout_mask_area, self.partial_fc_split, self.samples_per_mining = random_cutout_mask_area, partial_fc_split, samples_per_mining
#         self.train_ds, self.steps_per_epoch, self.classes, self.is_triplet_dataset = None, None, 0, False
#         self.teacher_model_interf, self.is_distill_ds = teacher_model_interf, False
#         self.distill_emb_map_layer = None
#         print("Initialization complete")  # Debug print

#     def __search_embedding_layer__(self, model):
#         for ii in range(1, 6):
#             if model.layers[-ii].name == "embedding":
#                 return -ii
    
#     def __init_dataset__(self, type, emb_loss_names):
#         init_as_triplet = self.triplet in emb_loss_names or type == self.triplet
#         is_offline_triplet = self.samples_per_mining > 0
#         if self.train_ds is not None and init_as_triplet == self.is_triplet_dataset and not self.is_distill_ds and not is_offline_triplet:
#             return

#         dataset_params = {
#             "data_path": self.data_path,
#             "batch_size": self.batch_size,
#             "random_status": self.random_status,
#             "random_cutout_mask_area": self.random_cutout_mask_area,
#             "image_per_class": self.image_per_class,
#             "mixup_alpha": self.mixup_alpha,
#             "teacher_model_interf": self.teacher_model_interf,
#         }
#         print(f"Initializing dataset with type: {type}, params: {dataset_params}")  # Debug print

#         if is_offline_triplet:
#             print(">>>> Init offline triplet dataset...")
#             aa = data.Triplet_dataset_offline(basic_model=self.basic_model, samples_per_mining=self.samples_per_mining, **dataset_params)
#             self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = False
#         elif init_as_triplet:
#             print(">>>> Init triplet dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 print(">>>> Combining tfrecord dataset with triplet is NOT recommended.")
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 aa = data.Triplet_dataset(**dataset_params)
#                 self.train_ds, self.steps_per_epoch = aa.ds, aa.steps_per_epoch
#             self.is_triplet_dataset = True
#         else:
#             print(">>>> Init softmax dataset...")
#             if self.data_path.endswith(".tfrecord"):
#                 self.train_ds, self.steps_per_epoch = data.prepare_distill_dataset_tfrecord(**dataset_params)
#             else:
#                 self.train_ds, self.steps_per_epoch = data.prepare_dataset(**dataset_params, partial_fc_split=self.partial_fc_split)
#             self.is_triplet_dataset = False
#         if self.train_ds is None:
#             print(">>>> [Error]: Dataset initialization failed, train_ds is None.")  # Debug print
#             return

#         if tf.distribute.has_strategy():
#             self.train_ds = self.train_ds.with_options(self.data_options)

#         label_spec = self.train_ds.element_spec[-1]
#         if isinstance(label_spec, tuple):
#             self.is_distill_ds = True
#             self.teacher_emb_size = label_spec[0].shape[-1]
#             self.classes = label_spec[1].shape[-1]
#             if type == self.distill:
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[1:] * len(emb_loss_names) + yy[:1]))
#             elif (self.distill in emb_loss_names and len(emb_loss_names) != 1) or (self.distill not in emb_loss_names and len(emb_loss_names) != 0):
#                 label_data_len = len(emb_loss_names) if self.distill in emb_loss_names else len(emb_loss_names) + 1
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[:1] + yy[1:] * label_data_len))
#         else:
#             self.is_distill_ds = False
#             self.classes = label_spec.shape[-1]
#         print(f"Dataset initialized: classes={self.classes}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
    
#     def __init_optimizer__(self, optimizer):
#         if optimizer == None:
#             if self.model != None and self.model.optimizer != None:
#                 self.optimizer = self.model.optimizerinstallation
#                 compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#                 print(">>>> Reuse optimizer from previous model:", compiled_opt.__class__.__name__)
#             else:
#                 print(">>>> Use default optimizer:", self.default_optimizer)
#                 self.optimizer = self.default_optimizer
#         else:
#             print(">>>> Use specified optimizer:", optimizer)
#             self.optimizer = optimizer

#         try:
#             import tensorflow_addons as tfa
#         except:
#             pass
#         else:
#             compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#             if isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
#                 print(">>>> Append weight decay callback...")
#                 lr_base, wd_base = self.optimizer.lr.numpy(), self.optimizer.weight_decay.numpy()
#                 wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=self.is_lr_on_batch)
#                 self.callbacks.append(wd_callback)

#     def __init_model__(self, type, loss_top_k=1, header_append_norm=False):
#         inputs = self.basic_model.inputs[0]
#         embedding = self.basic_model.outputs[0]
#         is_multi_output = lambda mm: len(mm.outputs) != 1 or isinstance(mm.layers[-1], keras.layers.Concatenate)
#         if self.model != None and is_multi_output(self.model):
#             output_layer = min(len(self.basic_model.layers), len(self.model.layers) - 1)
#             self.model = keras.models.Model(inputs, self.model.layers[output_layer].output)

#         if self.output_weight_decay != 0:
#             print(">>>> Add L2 regularizer to model output layer, output_weight_decay = %f" % self.output_weight_decay)
#             output_kernel_regularizer = keras.regularizers.L2(self.output_weight_decay / 2)
#         else:
#             output_kernel_regularizer = None

#         model_output_layer_name = None if self.model is None else self.model.output_names[-1]
#         if type == self.softmax and model_output_layer_name != self.softmax:
#             print(">>>> Add softmax layer...")
#             softmax_logits = keras.layers.Dense(self.classes, use_bias=False, name=self.softmax + "_logits", kernel_regularizer=output_kernel_regularizer)
#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 softmax_logits.build(embedding.shape)
#                 weight_cur = softmax_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     softmax_logits.set_weights(weight_pre)
#             logits = softmax_logits(embedding)
#             output_fp32 = keras.layers.Activation("softmax", dtype="float32", name=self.softmax)(logits)
#             self.model = keras.models.Model(inputs, output_fp32)
#         elif type == self.arcface and (model_output_layer_name != self.arcface or self.model.layers[-1].append_norm != header_append_norm):
#             vpl_start_iters = self.vpl_start_iters * self.steps_per_epoch if self.vpl_start_iters < 50 else self.vpl_start_iters
#             vpl_kwargs = {"vpl_lambda": 0.15, "start_iters": vpl_start_iters, "allowed_delta": self.vpl_allowed_delta}
#             arc_kwargs = {"loss_top_k": loss_top_k, "append_norm": header_append_norm, "partial_fc_split": self.partial_fc_split, "name": self.arcface}
#             print(">>>> Add arcface layer, arc_kwargs={}, vpl_kwargs={}...".format(arc_kwargs, vpl_kwargs))
#             if vpl_start_iters > 0:
#                 batch_size = self.batch_size_per_replica
#                 arcface_logits = models.NormDenseVPL(batch_size, self.classes, output_kernel_regularizer, **arc_kwargs, **vpl_kwargs, dtype="float32")
#             else:
#                 arcface_logits = models.NormDense(self.classes, output_kernel_regularizer, **arc_kwargs, dtype="float32")

#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 arcface_logits.build(embedding.shape)
#                 weight_cur = arcface_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     arcface_logits.set_weights(weight_pre)
#             output_fp32 = arcface_logits(embedding)
#             self.model = keras.models.Model(inputs, output_fp32)
#             # Print the summary of the full model (with ArcFace layer)
#             print(">>>> Printing full model summary (with ArcFace layer):")
#             self.model.summary()
#         elif type in [self.triplet, self.center, self.distill]:
#             self.model = self.basic_model
#             self.model.output_names[0] = type + "_embedding"
#         else:
#             print(">>>> Will NOT change model output layer.")

#         if self.pretrained is not None:
#             if self.model is None:
#                 self.basic_model.load_weights(self.pretrained)
#             else:
#                 self.model.load_weights(self.pretrained)
#             self.pretrained = None
#         print(f"Model initialization complete for type: {type}")  # Debug print

#     def __add_emb_output_to_model__(self, emb_type, emb_loss, emb_loss_weight):
#         nns = self.model.output_names
#         emb_shape = self.basic_model.output_shape[-1]
#         if emb_type == self.distill and self.teacher_emb_size != emb_shape:
#             print(">>>> Add a dense layer to map embedding: student %d --> teacher %d" % (emb_shape, self.teacher_emb_size))
#             embedding = self.basic_model.outputs[0]
#             if self.distill_emb_map_layer is None:
#                 self.distill_emb_map_layer = keras.layers.Dense(self.teacher_emb_size, use_bias=False, name="distill_map", dtype="float32")
#             emb_map_output = self.distill_emb_map_layer(embedding)
#             self.model = keras.models.Model(self.model.inputs[0], [emb_map_output] + self.model.outputs)
#         else:
#             self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)

#         self.model.output_names[0] = emb_type + "_embedding"
#         for id, nn in enumerate(nns):
#             self.model.output_names[id + 1] = nn
#         self.cur_loss, self.loss_weights = [emb_loss, *self.cur_loss], {ii: lossWeight for ii in self.model.output_names}
#         self.loss_weights.update({self.model.output_names[0]: emb_loss_weight})

#     def __init_type_by_loss__(self, loss):
#         print(">>>> Init type by loss function name...")
#         if isinstance(loss, str):
#             return self.softmax

#         if loss.__class__.__name__ == "function":
#             ss = loss.__name__.lower()
#             if self.softmax in ss:
#                 return self.softmax
#             if self.arcface in ss:
#                 return self.arcface
#             if self.triplet in ss:
#                 return self.triplet
#             if self.distill in ss:
#                 return self.distill
#         else:
#             ss = loss.__class__.__name__.lower()
#             if isinstance(loss, losses.TripletLossWapper) or self.triplet in ss:
#                 return self.triplet
#             if isinstance(loss, losses.CenterLoss) or self.center in ss:
#                 return self.center
#             if isinstance(loss, losses.ArcfaceLoss) or self.arcface in ss:
#                 return self.arcface
#             if isinstance(loss, losses.ArcfaceLossSimple) or isinstance(loss, losses.AdaCosLoss):
#                 return self.arcface
#             if isinstance(loss, losses.DistillKLDivergenceLoss):
#                 return self.arcface
#             if self.softmax in ss:
#                 return self.softmax
#         return self.softmax

#     def __init_emb_losses__(self, embLossTypes=None, embLossWeights=1):
#         emb_loss_names, emb_loss_weights = {}, {}
#         if embLossTypes is not None:
#             embLossTypes = embLossTypes if isinstance(embLossTypes, list) else [embLossTypes]
#             for id, ee in enumerate(embLossTypes):
#                 emb_loss_name = ee.lower() if isinstance(ee, str) else ee.__name__.lower()
#                 emb_loss_weight = float(embLossWeights[id] if isinstance(embLossWeights, list) else embLossWeights)
#                 if "centerloss" in emb_loss_name:
#                     emb_loss_names[self.center] = losses.CenterLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.center] = emb_loss_weight
#                 elif "triplet" in emb_loss_name:
#                     emb_loss_names[self.triplet] = losses.BatchHardTripletLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.triplet] = emb_loss_weight
#                 elif "distill" in emb_loss_name:
#                     emb_loss_names[self.distill] = losses.distiller_loss_cosine if ee == None or isinstance(ee, str) else ee
#                     emb_loss_weights[self.distill] = emb_loss_weight
#         return emb_loss_names, emb_loss_weights

#     def __basic_train__(self, epochs, initial_epoch=0):
#         self.model.compile(optimizer=self.optimizer, loss=self.cur_loss, metrics=self.metrics, loss_weights=self.loss_weights)
#         print(f"Starting model training: epochs={epochs}, initial_epoch={initial_epoch}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
#         self.model.fit(
#             self.train_ds,
#             epochs=epochs,
#             verbose=1,
#             callbacks=self.callbacks,
#             initial_epoch=initial_epoch,
#             steps_per_epoch=self.steps_per_epoch,
#             use_multiprocessing=True,
#             workers=4,
#         )
#         print("Training completed")  # Debug print

#     def reset_dataset(self, data_path=None):
#         self.train_ds = None
#         if data_path != None:
#             self.data_path = data_path

#     def train_single_scheduler(
#         self,
#         epoch,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35,
#     ):
#         emb_loss_names, emb_loss_weights = self.__init_emb_losses__(embLossTypes, embLossWeights)

#         # If no loss is provided, use a default ArcFace loss
#         if loss is None:
#             print(">>>> No loss specified, using default ArcFace loss...")
#             loss = losses.ArcfaceLoss()
#             if self.model is not None and self.model.built:
#                 print(">>>> Compiling model with default ArcFace loss...")
#                 self.model.compile(
#                     optimizer="adam",
#                     loss=loss,
#                     metrics=["accuracy"]
#                 )

#         if type is None and not self.inited_from_model:
#             type = self.__init_type_by_loss__(loss)
#         print(">>>> Train %s..." % type)
#         self.__init_dataset__(type, emb_loss_names)
#         if self.train_ds is None:
#             print(">>>> [Error]: train_ds is None.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return
#         if self.is_distill_ds == False and type == self.distill:
#             print(">>>> [Error]: Dataset doesn't contain embedding data.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return

#         self.is_lr_on_batch = isinstance(self.lr_scheduler, myCallbacks.CosineLrScheduler)
#         if self.is_lr_on_batch:
#             self.lr_scheduler.steps_per_epoch = self.steps_per_epoch

#         basic_callbacks = [ii for ii in [self.my_history, self.model_checkpoint, self.lr_scheduler] if ii is not None]
#         self.callbacks = self.my_evals + self.custom_callbacks + basic_callbacks
#         self.__init_optimizer__(optimizer)
#         if not self.inited_from_model:
#             header_append_norm = isinstance(loss, losses.MagFaceLoss) or isinstance(loss, losses.AdaFaceLoss)
#             self.__init_model__(type, lossTopK, header_append_norm)

#         self.cur_loss, self.loss_weights = [loss], {ii: lossWeight for ii in self.model.output_names}
#         if self.center in emb_loss_names and type != self.center:
#             loss_class = emb_loss_names[self.center]
#             print(">>>> Attach center loss:", loss_class.__name__)
#             emb_shape = self.basic_model.output_shape[-1]
#             initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
#             center_loss = loss_class(self.classes, emb_shape=emb_shape, initial_file=initial_file)
#             self.callbacks.append(center_loss.save_centers_callback)
#             self.__add_emb_output_to_model__(self.center, center_loss, emb_loss_weights[self.center])

#         if self.triplet in emb_loss_names and type != self.triplet:
#             loss_class = emb_loss_names[self.triplet]
#             print(">>>> Attach triplet loss: %s, alpha = %f..." % (loss_class.__name__, tripletAlpha))
#             triplet_loss = loss_class(alpha=tripletAlpha)
#             self.__add_emb_output_to_model__(self.triplet, triplet_loss, emb_loss_weights[self.triplet])

#         if self.is_distill_ds and type != self.distill:
#             distill_loss = emb_loss_names.get(self.distill, losses.distiller_loss_cosine)
#             print(">>>> Attach distill loss:", distill_loss.__name__)
#             self.__add_emb_output_to_model__(self.distill, distill_loss, emb_loss_weights.get(self.distill, 1))

#         print(">>>> loss_weights:", self.loss_weights)
#         self.metrics = {ii: None if "embedding" in ii else "accuracy" for ii in self.model.output_names}
#         self.callbacks.append(myCallbacks.ExitOnNaN())
#         if self.vpl_start_iters > 0:
#             loss.build(self.batch_size_per_replica)
#             self.callbacks.append(myCallbacks.VPLUpdateQueue())

#         if self.gently_stop:
#             self.callbacks.append(self.gently_stop)

#         if bottleneckOnly:
#             print(">>>> Train bottleneckOnly...")
#             self.basic_model.trainable = False
#             self.callbacks = self.callbacks[len(self.my_evals) :]
#             self.__basic_train__(epoch, initial_epoch=0)
#             self.basic_model.trainable = True
#         else:
#             self.__basic_train__(initial_epoch + epoch, initial_epoch=initial_epoch)

#         print(">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s" % (type, self.model.history.epoch, self.model.stop_training))
#         print(">>>> My history:")
#         self.my_history.print_hist()
#         latest_save_path = os.path.join("checkpoints", os.path.splitext(self.save_path)[0] + "_basic_model_latest.h5")
#         print(">>>> Saving latest basic model to:", latest_save_path)
#         self.basic_model.save(latest_save_path)

#     def train(self, train_schedule, initial_epoch=0):
#         train_schedule = [train_schedule] if isinstance(train_schedule, dict) else train_schedule
#         for sch in train_schedule:
#             for ii in ["centerloss", "triplet", "distill"]:
#                 if ii in sch:
#                     sch.setdefault("embLossTypes", []).append(ii)
#                     sch.setdefault("embLossWeights", []).append(sch.pop(ii))
#             if "alpha" in sch:
#                 sch["tripletAlpha"] = sch.pop("alpha")

#             self.train_single_scheduler(**sch, initial_epoch=initial_epoch)
#             initial_epoch += 0 if sch.get("bottleneckOnly", False) else sch["epoch"]

#             if self.model is None or self.model.stop_training == True:
#                 print(">>>> But it's an early stop, break...")
#                 break
#         return initial_epoch

# if __name__ == "__main__":
#     print("Entering main execution block")  # Debug print
#     # Prompt user for dataset directory at runtime
#     data_dir = input("Please enter the dataset directory path: ")
#     print(f"Dataset directory provided: {data_dir}")  # Debug print
#     trainer = Train(
#         data_path=data_dir,
#         save_path="train",
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     )
#     print("Train instance created")  # Debug print
#     trainer.train_single_scheduler(
#         epoch=10,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35
#     )








#taake multiple paths as input ids int
# import sys
# sys.path.append("C:\\Users\\HP\\Documents\\zmine\\parttime\\ghostfacerepo\\projectrepo\\GhostFaceNets")

# import os
# print("Imported os")  # Debug print
# import data
# print("Imported data")  # Debug print
# import evals
# print("Imported evals")  # Debug print
# import losses
# print("Imported losses")  # Debug print
# import GhostFaceNets, GhostFaceNets_with_Bias
# print("Imported GhostFaceNets modules")  # Debug print
# import myCallbacks
# print("Imported myCallbacks")  # Debug print
# import tensorflow as tf
# print("Imported tensorflow")  # Debug print
# from tensorflow import keras
# print("Imported keras")  # Debug print
# import models
# print("Imported models")  # Debug print
# import argparse
# print("Imported argparse")  # Debug print

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# print("Starting script execution")  # Debug print

# class Train:
#     def __init__(
#         self,
#         data_paths,  # Changed from data_path to data_paths to accept a list
#         save_path,
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     ):
#         print("Entered Train.__init__")  # Debug print
#         from inspect import getmembers, isfunction, isclass

#         custom_objects.update(dict([ii for ii in getmembers(losses) if isfunction(ii[1]) or isclass(ii[1])]))
#         custom_objects.update({"NormDense": models.NormDense})
#         print("Custom objects updated")  # Debug print

#         self.model, self.basic_model, self.save_path, self.inited_from_model, self.sam_rho, self.pretrained = None, None, save_path, False, sam_rho, pretrained
#         self.vpl_start_iters, self.vpl_allowed_delta = vpl_start_iters, vpl_allowed_delta
#         print("Instance variables initialized")  # Debug print
        
#         # Initialize a new model with ghostnetv2 backbone
#         print("Initializing new model with ghostnetv2 backbone")
#         self.basic_model = GhostFaceNets.buildin_models(
#             stem_model="ghostnetv2",
#             input_shape=(112, 112, 3),
#             dropout=0,
#             emb_shape=512,
#             output_layer="GDC",
#             bn_momentum=0.99,
#             bn_epsilon=0.001,
#             weights=None  # No pre-trained weights
#         )
#         print("New model initialized as basic_model using ghostnetv2 backbone")
#         # Print the summary of the basic model (includes FocalModulationBlock)
#         print(">>>> Printing basic_model summary (includes FocalModulationBlock):")
#         self.basic_model.summary()

#         if self.basic_model is None:
#             print(
#                 "Initialize model by:\n"
#                 "| basic_model                                                     | model           |\n"
#                 "| --------------------------------------------------------------- | --------------- |\n"
#                 "| model structure                                                 | None            |\n"
#                 "| basic model .h5 file                                            | None            |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
#                 "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
#                 "| None                                                            | None            |\n"
#                 "* Both None for reload model from 'checkpoints/{}'\n".format(save_path)
#             )
#             return

#         # Losses
#         self.softmax, self.arcface, self.arcface_partial, self.triplet = "softmax", "arcface", "arcface_partial", "triplet"
#         self.center, self.distill = "center", "distill"
        
#         if output_weight_decay >= 1:
#             l2_weight_decay = 0
#             for ii in self.basic_model.layers:
#                 if hasattr(ii, "kernel_regularizer") and isinstance(ii.kernel_regularizer, keras.regularizers.L2):
#                     l2_weight_decay = ii.kernel_regularizer.l2
#                     break
#             print(">>>> L2 regularizer value from basic_model:", l2_weight_decay)
#             output_weight_decay *= l2_weight_decay * 2
#         self.output_weight_decay = output_weight_decay

#         self.batch_size, self.batch_size_per_replica = batch_size, batch_size
#         if tf.distribute.has_strategy():
#             strategy = tf.distribute.get_strategy()
#             self.batch_size = batch_size * strategy.num_replicas_in_sync
#             print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
#             self.data_options = tf.data.Options()
#             self.data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
#         # Evaluate
#         my_evals = [evals.eval_callback(self.basic_model, ii, batch_size=self.batch_size_per_replica, eval_freq=eval_freq) for ii in eval_paths]
#         if len(my_evals) != 0:
#             my_evals[-1].save_model = os.path.splitext(save_path)[0]
        
#         self.my_history, self.model_checkpoint, self.lr_scheduler, self.gently_stop = myCallbacks.basic_callbacks(
#             save_path,
#             my_evals,
#             lr=lr_base,
#             lr_decay=lr_decay,
#             lr_min=lr_min,
#             lr_decay_steps=lr_decay_steps,
#             lr_warmup_steps=lr_warmup_steps,
#         )
#         self.gently_stop = None  # may not working for windows
#         self.my_evals, self.custom_callbacks = my_evals, []
#         self.metrics = ["accuracy"]
#         self.default_optimizer = "adam"

#         self.data_paths, self.random_status, self.image_per_class, self.mixup_alpha = data_paths, random_status, image_per_class, mixup_alpha
#         self.random_cutout_mask_area, self.partial_fc_split, self.samples_per_mining = random_cutout_mask_area, partial_fc_split, samples_per_mining
#         self.train_ds, self.steps_per_epoch, self.classes, self.is_triplet_dataset = None, None, 0, False
#         self.teacher_model_interf, self.is_distill_ds = teacher_model_interf, False
#         self.distill_emb_map_layer = None
#         print("Initialization complete")  # Debug print

#     def __search_embedding_layer__(self, model):
#         for ii in range(1, 6):
#             if model.layers[-ii].name == "embedding":
#                 return -ii
    
#     def __init_dataset__(self, type, emb_loss_names):
#         init_as_triplet = self.triplet in emb_loss_names or type == self.triplet
#         is_offline_triplet = self.samples_per_mining > 0
#         if self.train_ds is not None and init_as_triplet == self.is_triplet_dataset and not self.is_distill_ds and not is_offline_triplet:
#             return

#         dataset_params = {
#             "batch_size": self.batch_size,
#             "random_status": self.random_status,
#             "random_cutout_mask_area": self.random_cutout_mask_area,
#             "image_per_class": self.image_per_class,
#             "mixup_alpha": self.mixup_alpha,
#             "teacher_model_interf": self.teacher_model_interf,
#         }
#         print(f"Initializing datasets with type: {type}, params: {dataset_params}")  # Debug print

#         # Initialize datasets for each path and combine them
#         combined_ds = None
#         total_steps_per_epoch = 0
#         total_classes = None

#         for data_path in self.data_paths:
#             print(f">>>> Processing dataset path: {data_path}")
#             dataset_params["data_path"] = data_path

#             if is_offline_triplet:
#                 print(">>>> Init offline triplet dataset...")
#                 aa = data.Triplet_dataset_offline(basic_model=self.basic_model, samples_per_mining=self.samples_per_mining, **dataset_params)
#                 ds, steps = aa.ds, aa.steps_per_epoch
#                 is_triplet = False
#             elif init_as_triplet:
#                 print(">>>> Init triplet dataset...")
#                 if data_path.endswith(".tfrecord"):
#                     print(">>>> Combining tfrecord dataset with triplet is NOT recommended.")
#                     ds, steps = data.prepare_distill_dataset_tfrecord(**dataset_params)
#                 else:
#                     aa = data.Triplet_dataset(**dataset_params)
#                     ds, steps = aa.ds, aa.steps_per_epoch
#                 is_triplet = True
#             else:
#                 print(">>>> Init softmax dataset...")
#                 if data_path.endswith(".tfrecord"):
#                     ds, steps = data.prepare_distill_dataset_tfrecord(**dataset_params)
#                 else:
#                     ds, steps = data.prepare_dataset(**dataset_params, partial_fc_split=self.partial_fc_split)
#                 is_triplet = False

#             if ds is None:
#                 print(f">>>> [Error]: Dataset initialization failed for path {data_path}, ds is None.")
#                 return

#             # Combine datasets
#             if combined_ds is None:
#                 combined_ds = ds
#                 total_steps_per_epoch = steps
#                 self.is_triplet_dataset = is_triplet
#             else:
#                 combined_ds = combined_ds.concatenate(ds)
#                 total_steps_per_epoch += steps
#                 if is_triplet != self.is_triplet_dataset:
#                     print(f">>>> [Warning]: Inconsistent dataset types (triplet vs non-triplet) across paths. Using {type}.")
#                     self.is_triplet_dataset = is_triplet

#             # Check classes for consistency
#             label_spec = ds.element_spec[-1]
#             if isinstance(label_spec, tuple):
#                 classes = label_spec[1].shape[-1]
#             else:
#                 classes = label_spec.shape[-1]
            
#             if total_classes is None:
#                 total_classes = classes
#             elif total_classes != classes:
#                 print(f">>>> [Warning]: Inconsistent number of classes ({total_classes} vs {classes}) in path {data_path}. Using {total_classes}.")

#         if combined_ds is None:
#             print(">>>> [Error]: No valid datasets initialized, combined_ds is None.")
#             return

#         self.train_ds, self.steps_per_epoch = combined_ds, total_steps_per_epoch
#         self.classes = total_classes

#         if tf.distribute.has_strategy():
#             self.train_ds = self.train_ds.with_options(self.data_options)

#         label_spec = self.train_ds.element_spec[-1]
#         if isinstance(label_spec, tuple):
#             self.is_distill_ds = True
#             self.teacher_emb_size = label_spec[0].shape[-1]
#             self.classes = label_spec[1].shape[-1]
#             if type == self.distill:
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[1:] * len(emb_loss_names) + yy[:1]))
#             elif (self.distill in emb_loss_names and len(emb_loss_names) != 1) or (self.distill not in emb_loss_names and len(emb_loss_names) != 0):
#                 label_data_len = len(emb_loss_names) if self.distill in emb_loss_names else len(emb_loss_names) + 1
#                 self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[:1] + yy[1:] * label_data_len))
#         else:
#             self.is_distill_ds = False
#             self.classes = label_spec.shape[-1]
#         print(f"Dataset initialized: classes={self.classes}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
    
#     def __init_optimizer__(self, optimizer):
#         if optimizer == None:
#             if self.model != None and self.model.optimizer != None:
#                 self.optimizer = self.model.optimizer
#                 compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#                 print(">>>> Reuse optimizer from previous model:", compiled_opt.__class__.__name__)
#             else:
#                 print(">>>> Use default optimizer:", self.default_optimizer)
#                 self.optimizer = self.default_optimizer
#         else:
#             print(">>>> Use specified optimizer:", optimizer)
#             self.optimizer = optimizer

#         try:
#             import tensorflow_addons as tfa
#         except:
#             pass
#         else:
#             compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
#             if isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
#                 print(">>>> Append weight decay callback...")
#                 lr_base, wd_base = self.optimizer.lr.numpy(), self.optimizer.weight_decay.numpy()
#                 wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=self.is_lr_on_batch)
#                 self.callbacks.append(wd_callback)

#     def __init_model__(self, type, loss_top_k=1, header_append_norm=False):
#         inputs = self.basic_model.inputs[0]
#         embedding = self.basic_model.outputs[0]
#         is_multi_output = lambda mm: len(mm.outputs) != 1 or isinstance(mm.layers[-1], keras.layers.Concatenate)
#         if self.model != None and is_multi_output(self.model):
#             output_layer = min(len(self.basic_model.layers), len(self.model.layers) - 1)
#             self.model = keras.models.Model(inputs, self.model.layers[output_layer].output)

#         if self.output_weight_decay != 0:
#             print(">>>> Add L2 regularizer to model output layer, output_weight_decay = %f" % self.output_weight_decay)
#             output_kernel_regularizer = keras.regularizers.L2(self.output_weight_decay / 2)
#         else:
#             output_kernel_regularizer = None

#         model_output_layer_name = None if self.model is None else self.model.output_names[-1]
#         if type == self.softmax and model_output_layer_name != self.softmax:
#             print(">>>> Add softmax layer...")
#             softmax_logits = keras.layers.Dense(self.classes, use_bias=False, name=self.softmax + "_logits", kernel_regularizer=output_kernel_regularizer)
#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 softmax_logits.build(embedding.shape)
#                 weight_cur = softmax_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     softmax_logits.set_weights(weight_pre)
#             logits = softmax_logits(embedding)
#             output_fp32 = keras.layers.Activation("softmax", dtype="float32", name=self.softmax)(logits)
#             self.model = keras.models.Model(inputs, output_fp32)
#         elif type == self.arcface and (model_output_layer_name != self.arcface or self.model.layers[-1].append_norm != header_append_norm):
#             vpl_start_iters = self.vpl_start_iters * self.steps_per_epoch if self.vpl_start_iters < 50 else self.vpl_start_iters
#             vpl_kwargs = {"vpl_lambda": 0.15, "start_iters": vpl_start_iters, "allowed_delta": self.vpl_allowed_delta}
#             arc_kwargs = {"loss_top_k": loss_top_k, "append_norm": header_append_norm, "partial_fc_split": self.partial_fc_split, "name": self.arcface}
#             print(">>>> Add arcface layer, arc_kwargs={}, vpl_kwargs={}...".format(arc_kwargs, vpl_kwargs))
#             if vpl_start_iters > 0:
#                 batch_size = self.batch_size_per_replica
#                 arcface_logits = models.NormDenseVPL(batch_size, self.classes, output_kernel_regularizer, **arc_kwargs, **vpl_kwargs, dtype="float32")
#             else:
#                 arcface_logits = models.NormDense(self.classes, output_kernel_regularizer, **arc_kwargs, dtype="float32")

#             if self.model != None and "_embedding" not in self.model.output_names[-1]:
#                 arcface_logits.build(embedding.shape)
#                 weight_cur = arcface_logits.get_weights()
#                 weight_pre = self.model.layers[-1].get_weights()
#                 if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
#                     print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
#                     arcface_logits.set_weights(weight_pre)
#             output_fp32 = arcface_logits(embedding)
#             self.model = keras.models.Model(inputs, output_fp32)
#             # Print the summary of the full model (with ArcFace layer)
#             print(">>>> Printing full model summary (with ArcFace layer):")
#             self.model.summary()
#         elif type in [self.triplet, self.center, self.distill]:
#             self.model = self.basic_model
#             self.model.output_names[0] = type + "_embedding"
#         else:
#             print(">>>> Will NOT change model output layer.")

#         if self.pretrained is not None:
#             if self.model is None:
#                 self.basic_model.load_weights(self.pretrained)
#             else:
#                 self.model.load_weights(self.pretrained)
#             self.pretrained = None
#         print(f"Model initialization complete for type: {type}")  # Debug print

#     def __add_emb_output_to_model__(self, emb_type, emb_loss, emb_loss_weight):
#         nns = self.model.output_names
#         emb_shape = self.basic_model.output_shape[-1]
#         if emb_type == self.distill and self.teacher_emb_size != emb_shape:
#             print(">>>> Add a dense layer to map embedding: student %d --> teacher %d" % (emb_shape, self.teacher_emb_size))
#             embedding = self.basic_model.outputs[0]
#             if self.distill_emb_map_layer is None:
#                 self.distill_emb_map_layer = keras.layers.Dense(self.teacher_emb_size, use_bias=False, name="distill_map", dtype="float32")
#             emb_map_output = self.distill_emb_map_layer(embedding)
#             self.model = keras.models.Model(self.model.inputs[0], [emb_map_output] + self.model.outputs)
#         else:
#             self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)

#         self.model.output_names[0] = emb_type + "_embedding"
#         for id, nn in enumerate(nns):
#             self.model.output_names[id + 1] = nn
#         self.cur_loss, self.loss_weights = [emb_loss, *self.cur_loss], {ii: lossWeight for ii in self.model.output_names}
#         self.loss_weights.update({self.model.output_names[0]: emb_loss_weight})

#     def __init_type_by_loss__(self, loss):
#         print(">>>> Init type by loss function name...")
#         if isinstance(loss, str):
#             return self.softmax

#         if loss.__class__.__name__ == "function":
#             ss = loss.__name__.lower()
#             if self.softmax in ss:
#                 return self.softmax
#             if self.arcface in ss:
#                 return self.arcface
#             if self.triplet in ss:
#                 return self.triplet
#             if self.distill in ss:
#                 return self.distill
#         else:
#             ss = loss.__class__.__name__.lower()
#             if isinstance(loss, losses.TripletLossWapper) or self.triplet in ss:
#                 return self.triplet
#             if isinstance(loss, losses.CenterLoss) or self.center in ss:
#                 return self.center
#             if isinstance(loss, losses.ArcfaceLoss) or self.arcface in ss:
#                 return self.arcface
#             if isinstance(loss, losses.ArcfaceLossSimple) or isinstance(loss, losses.AdaCosLoss):
#                 return self.arcface
#             if isinstance(loss, losses.DistillKLDivergenceLoss):
#                 return self.arcface
#             if self.softmax in ss:
#                 return self.softmax
#         return self.softmax

#     def __init_emb_losses__(self, embLossTypes=None, embLossWeights=1):
#         emb_loss_names, emb_loss_weights = {}, {}
#         if embLossTypes is not None:
#             embLossTypes = embLossTypes if isinstance(embLossTypes, list) else [embLossTypes]
#             for id, ee in enumerate(embLossTypes):
#                 emb_loss_name = ee.lower() if isinstance(ee, str) else ee.__name__.lower()
#                 emb_loss_weight = float(embLossWeights[id] if isinstance(embLossWeights, list) else embLossWeights)
#                 if "centerloss" in emb_loss_name:
#                     emb_loss_names[self.center] = losses.CenterLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.center] = emb_loss_weight
#                 elif "triplet" in emb_loss_name:
#                     emb_loss_names[self.triplet] = losses.BatchHardTripletLoss if isinstance(ee, str) else ee
#                     emb_loss_weights[self.triplet] = emb_loss_weight
#                 elif "distill" in emb_loss_name:
#                     emb_loss_names[self.distill] = losses.distiller_loss_cosine if ee == None or isinstance(ee, str) else ee
#                     emb_loss_weights[self.distill] = emb_loss_weight
#         return emb_loss_names, emb_loss_weights

#     def __basic_train__(self, epochs, initial_epoch=0):
#         self.model.compile(optimizer=self.optimizer, loss=self.cur_loss, metrics=self.metrics, loss_weights=self.loss_weights)
#         print(f"Starting model training: epochs={epochs}, initial_epoch={initial_epoch}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
#         self.model.fit(
#             self.train_ds,
#             epochs=epochs,
#             verbose=1,
#             callbacks=self.callbacks,
#             initial_epoch=initial_epoch,
#             steps_per_epoch=self.steps_per_epoch,
#             use_multiprocessing=True,
#             workers=4,
#         )
#         print("Training completed")  # Debug print

#     def reset_dataset(self, data_paths=None):  # Updated to accept data_paths
#         self.train_ds = None
#         if data_paths != None:
#             self.data_paths = data_paths

#     def train_single_scheduler(
#         self,
#         epoch,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35,
#     ):
#         emb_loss_names, emb_loss_weights = self.__init_emb_losses__(embLossTypes, embLossWeights)

#         # If no loss is provided, use a default ArcFace loss
#         if loss is None:
#             print(">>>> No loss specified, using default ArcFace loss...")
#             loss = losses.ArcfaceLoss()
#             if self.model is not None and self.model.built:
#                 print(">>>> Compiling model with default ArcFace loss...")
#                 self.model.compile(
#                     optimizer="adam",
#                     loss=loss,
#                     metrics=["accuracy"]
#                 )

#         if type is None and not self.inited_from_model:
#             type = self.__init_type_by_loss__(loss)
#         print(">>>> Train %s..." % type)
#         self.__init_dataset__(type, emb_loss_names)
#         if self.train_ds is None:
#             print(">>>> [Error]: train_ds is None.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return
#         if self.is_distill_ds == False and type == self.distill:
#             print(">>>> [Error]: Dataset doesn't contain embedding data.")
#             if self.model is not None:
#                 self.model.stop_training = True
#             return

#         self.is_lr_on_batch = isinstance(self.lr_scheduler, myCallbacks.CosineLrScheduler)
#         if self.is_lr_on_batch:
#             self.lr_scheduler.steps_per_epoch = self.steps_per_epoch

#         basic_callbacks = [ii for ii in [self.my_history, self.model_checkpoint, self.lr_scheduler] if ii is not None]
#         self.callbacks = self.my_evals + self.custom_callbacks + basic_callbacks
#         self.__init_optimizer__(optimizer)
#         if not self.inited_from_model:
#             header_append_norm = isinstance(loss, losses.MagFaceLoss) or isinstance(loss, losses.AdaFaceLoss)
#             self.__init_model__(type, lossTopK, header_append_norm)

#         self.cur_loss, self.loss_weights = [loss], {ii: lossWeight for ii in self.model.output_names}
#         if self.center in emb_loss_names and type != self.center:
#             loss_class = emb_loss_names[self.center]
#             print(">>>> Attach center loss:", loss_class.__name__)
#             emb_shape = self.basic_model.output_shape[-1]
#             initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
#             center_loss = loss_class(self.classes, emb_shape=emb_shape, initial_file=initial_file)
#             self.callbacks.append(center_loss.save_centers_callback)
#             self.__add_emb_output_to_model__(self.center, center_loss, emb_loss_weights[self.center])

#         if self.triplet in emb_loss_names and type != self.triplet:
#             loss_class = emb_loss_names[self.triplet]
#             print(">>>> Attach triplet loss: %s, alpha = %f..." % (loss_class.__name__, tripletAlpha))
#             triplet_loss = loss_class(alpha=tripletAlpha)
#             self.__add_emb_output_to_model__(self.triplet, triplet_loss, emb_loss_weights[self.triplet])

#         if self.is_distill_ds and type != self.distill:
#             distill_loss = emb_loss_names.get(self.distill, losses.distiller_loss_cosine)
#             print(">>>> Attach distill loss:", distill_loss.__name__)
#             self.__add_emb_output_to_model__(self.distill, distill_loss, emb_loss_weights.get(self.distill, 1))

#         print(">>>> loss_weights:", self.loss_weights)
#         self.metrics = {ii: None if "embedding" in ii else "accuracy" for ii in self.model.output_names}
#         self.callbacks.append(myCallbacks.ExitOnNaN())
#         if self.vpl_start_iters > 0:
#             loss.build(self.batch_size_per_replica)
#             self.callbacks.append(myCallbacks.VPLUpdateQueue())

#         if self.gently_stop:
#             self.callbacks.append(self.gently_stop)

#         if bottleneckOnly:
#             print(">>>> Train bottleneckOnly...")
#             self.basic_model.trainable = False
#             self.callbacks = self.callbacks[len(self.my_evals) :]
#             self.__basic_train__(epoch, initial_epoch=0)
#             self.basic_model.trainable = True
#         else:
#             self.__basic_train__(initial_epoch + epoch, initial_epoch=initial_epoch)

#         print(">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s" % (type, self.model.history.epoch, self.model.stop_training))
#         print(">>>> My history:")
#         self.my_history.print_hist()
#         latest_save_path = os.path.join("checkpoints", os.path.splitext(self.save_path)[0] + "_basic_model_latest.h5")
#         print(">>>> Saving latest basic model to:", latest_save_path)
#         self.basic_model.save(latest_save_path)

#     def train(self, train_schedule, initial_epoch=0):
#         train_schedule = [train_schedule] if isinstance(train_schedule, dict) else train_schedule
#         for sch in train_schedule:
#             for ii in ["centerloss", "triplet", "distill"]:
#                 if ii in sch:
#                     sch.setdefault("embLossTypes", []).append(ii)
#                     sch.setdefault("embLossWeights", []).append(sch.pop(ii))
#             if "alpha" in sch:
#                 sch["tripletAlpha"] = sch.pop("alpha")

#             self.train_single_scheduler(**sch, initial_epoch=initial_epoch)
#             initial_epoch += 0 if sch.get("bottleneckOnly", False) else sch["epoch"]

#             if self.model is None or self.model.stop_training == True:
#                 print(">>>> But it's an early stop, break...")
#                 break
#         return initial_epoch

# if __name__ == "__main__":
#     print("Entering main execution block")  # Debug print
#     # Prompt user for multiple dataset directories at runtime
#     data_paths = []
#     while True:
#         data_path = input("Please enter a dataset directory path (or press Enter to finish): ").strip()
#         if data_path == "":
#             break
#         data_paths.append(data_path)
    
#     if not data_paths:
#         print(">>>> [Error]: No dataset paths provided. Exiting.")
#         sys.exit(1)
    
#     print(f"Dataset directories provided: {data_paths}")  # Debug print
#     trainer = Train(
#         data_paths=data_paths,  # Pass list of paths
#         save_path="train",
#         eval_paths=[],
#         basic_model=None,
#         model=None,
#         compile=True,
#         output_weight_decay=1,
#         custom_objects={},
#         pretrained=None,
#         batch_size=8,
#         lr_base=0.001,
#         lr_decay=0.05,
#         lr_decay_steps=0,
#         lr_min=1e-6,
#         lr_warmup_steps=0,
#         eval_freq=1,
#         random_status=0,
#         random_cutout_mask_area=0.0,
#         image_per_class=0,
#         samples_per_mining=0,
#         mixup_alpha=0,
#         partial_fc_split=0,
#         teacher_model_interf=None,
#         sam_rho=0,
#         vpl_start_iters=-1,
#         vpl_allowed_delta=200,
#     )
#     print("Train instance created")  # Debug print
#     trainer.train_single_scheduler(
#         epoch=10,
#         loss=None,
#         initial_epoch=0,
#         lossWeight=1,
#         optimizer=None,
#         bottleneckOnly=False,
#         lossTopK=1,
#         type=None,
#         embLossTypes=None,
#         embLossWeights=1,
#         tripletAlpha=0.35
#     )



















import sys
sys.path.append("C:\\Users\\HP\\Documents\\zmine\\parttime\\ghostfacerepo\\projectrepo\\GhostFaceNets")

import os
print("Imported os")  # Debug print
import data
print("Imported data")  # Debug print
import evals
print("Imported evals")  # Debug print
import losses
print("Imported losses")  # Debug print
import GhostFaceNets, GhostFaceNets_with_Bias
print("Imported GhostFaceNets modules")  # Debug print
import myCallbacks
print("Imported myCallbacks")  # Debug print
import tensorflow as tf
print("Imported tensorflow")  # Debug print
from tensorflow import keras
print("Imported keras")  # Debug print
import models
print("Imported models")  # Debug print
import argparse
print("Imported argparse")  # Debug print

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Starting script execution")  # Debug print

class Train:
    def __init__(
        self,
        data_paths,  # Changed from data_path to data_paths to accept a list
        save_path,
        eval_paths=[],
        basic_model=None,
        model=None,
        compile=True,
        output_weight_decay=1,
        custom_objects={},
        pretrained=None,
        batch_size=8,
        lr_base=0.001,
        lr_decay=0.05,
        lr_decay_steps=0,
        lr_min=1e-6,
        lr_warmup_steps=0,
        eval_freq=1,
        random_status=0,
        random_cutout_mask_area=0.0,
        image_per_class=0,
        samples_per_mining=0,
        mixup_alpha=0,
        partial_fc_split=0,
        teacher_model_interf=None,
        sam_rho=0,
        vpl_start_iters=-1,
        vpl_allowed_delta=200,
    ):
        print("Entered Train.__init__")  # Debug print
        from inspect import getmembers, isfunction, isclass

        custom_objects.update(dict([ii for ii in getmembers(losses) if isfunction(ii[1]) or isclass(ii[1])]))
        custom_objects.update({"NormDense": models.NormDense})
        print("Custom objects updated")  # Debug print

        self.model, self.basic_model, self.save_path, self.inited_from_model, self.sam_rho, self.pretrained = None, None, save_path, False, sam_rho, pretrained
        self.vpl_start_iters, self.vpl_allowed_delta = vpl_start_iters, vpl_allowed_delta
        print("Instance variables initialized")  # Debug print
        
        # Initialize a new model with ghostnetv2 backbone
        print("Initializing new model with ghostnetv2 backbone")
        self.basic_model = GhostFaceNets.buildin_models(
            stem_model="ghostnetv2",
            input_shape=(112, 112, 3),
            dropout=0,
            emb_shape=512,
            output_layer="GDC",
            bn_momentum=0.99,
            bn_epsilon=0.001,
            weights=None  # No pre-trained weights
        )
        print("New model initialized as basic_model using ghostnetv2 backbone")
        # Print the summary of the basic model (includes FocalModulationBlock)
        print(">>>> Printing basic_model summary (includes FocalModulationBlock):")
        self.basic_model.summary()

        if self.basic_model is None:
            print(
                "Initialize model by:\n"
                "| basic_model                                                     | model           |\n"
                "| --------------------------------------------------------------- | --------------- |\n"
                "| model structure                                                 | None            |\n"
                "| basic model .h5 file                                            | None            |\n"
                "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
                "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
                "| None                                                            | None            |\n"
                "* Both None for reload model from 'checkpoints/{}'\n".format(save_path)
            )
            return

        # Losses
        self.softmax, self.arcface, self.arcface_partial, self.triplet = "softmax", "arcface", "arcface_partial", "triplet"
        self.center, self.distill = "center", "distill"
        
        if output_weight_decay >= 1:
            l2_weight_decay = 0
            for ii in self.basic_model.layers:
                if hasattr(ii, "kernel_regularizer") and isinstance(ii.kernel_regularizer, keras.regularizers.L2):
                    l2_weight_decay = ii.kernel_regularizer.l2
                    break
            print(">>>> L2 regularizer value from basic_model:", l2_weight_decay)
            output_weight_decay *= l2_weight_decay * 2
        self.output_weight_decay = output_weight_decay

        self.batch_size, self.batch_size_per_replica = batch_size, batch_size
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            self.batch_size = batch_size * strategy.num_replicas_in_sync
            print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
            self.data_options = tf.data.Options()
            self.data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
        # Evaluate
        my_evals = [evals.eval_callback(self.basic_model, ii, batch_size=self.batch_size_per_replica, eval_freq=eval_freq) for ii in eval_paths]
        if len(my_evals) != 0:
            my_evals[-1].save_model = os.path.splitext(save_path)[0]
        
        self.my_history, self.model_checkpoint, self.lr_scheduler, self.gently_stop = myCallbacks.basic_callbacks(
            save_path,
            my_evals,
            lr=lr_base,
            lr_decay=lr_decay,
            lr_min=lr_min,
            lr_decay_steps=lr_decay_steps,
            lr_warmup_steps=lr_warmup_steps,
        )
        self.gently_stop = None  # may not working for windows
        self.my_evals, self.custom_callbacks = my_evals, []
        self.metrics = ["accuracy"]
        self.default_optimizer = "adam"

        self.data_paths, self.random_status, self.image_per_class, self.mixup_alpha = data_paths, random_status, image_per_class, mixup_alpha
        self.random_cutout_mask_area, self.partial_fc_split, self.samples_per_mining = random_cutout_mask_area, partial_fc_split, samples_per_mining
        self.train_ds, self.steps_per_epoch, self.classes, self.is_triplet_dataset = None, None, 0, False
        self.teacher_model_interf, self.is_distill_ds = teacher_model_interf, False
        self.distill_emb_map_layer = None
        print("Initialization complete")  # Debug print

    def __search_embedding_layer__(self, model):
        for ii in range(1, 6):
            if model.layers[-ii].name == "embedding":
                return -ii
    
    def __init_dataset__(self, type, emb_loss_names):
        init_as_triplet = self.triplet in emb_loss_names or type == self.triplet
        is_offline_triplet = self.samples_per_mining > 0
        if self.train_ds is not None and init_as_triplet == self.is_triplet_dataset and not self.is_distill_ds and not is_offline_triplet:
            return

        dataset_params = {
            "batch_size": self.batch_size,
            "random_status": self.random_status,
            "random_cutout_mask_area": self.random_cutout_mask_area,
            "image_per_class": self.image_per_class,
            "mixup_alpha": self.mixup_alpha,
            "teacher_model_interf": self.teacher_model_interf,
        }
        print(f"Initializing datasets with type: {type}, params: {dataset_params}")  # Debug print

        # Initialize datasets for each path and combine them
        combined_ds = None
        total_steps_per_epoch = 0
        total_classes = None
        total_folders = 0  # To track total folders/classes across all paths

        for data_path in self.data_paths:
            print(f">>>> Picking data from path: {data_path}")  # Added print statement
            dataset_params["data_path"] = data_path

            # Count number of folders in the current path (assuming each folder is a class)
            if not data_path.endswith(".tfrecord") and os.path.exists(data_path):
                num_folders = len([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
                print(f">>>> Number of folders/classes in path {data_path}: {num_folders}")  # Added print statement
                total_folders += num_folders
            else:
                print(f">>>> Path {data_path} is a tfrecord or does not exist, skipping folder count.")

            if is_offline_triplet:
                print(">>>> Init offline triplet dataset...")
                aa = data.Triplet_dataset_offline(basic_model=self.basic_model, samples_per_mining=self.samples_per_mining, **dataset_params)
                ds, steps = aa.ds, aa.steps_per_epoch
                is_triplet = False
            elif init_as_triplet:
                print(">>>> Init triplet dataset...")
                if data_path.endswith(".tfrecord"):
                    print(">>>> Combining tfrecord dataset with triplet is NOT recommended.")
                    ds, steps = data.prepare_distill_dataset_tfrecord(**dataset_params)
                else:
                    aa = data.Triplet_dataset(**dataset_params)
                    ds, steps = aa.ds, aa.steps_per_epoch
                is_triplet = True
            else:
                print(">>>> Init softmax dataset...")
                if data_path.endswith(".tfrecord"):
                    ds, steps = data.prepare_distill_dataset_tfrecord(**dataset_params)
                else:
                    ds, steps = data.prepare_dataset(**dataset_params, partial_fc_split=self.partial_fc_split)
                is_triplet = False

            if ds is None:
                print(f">>>> [Error]: Dataset initialization failed for path {data_path}, ds is None.")
                return

            # Combine datasets
            if combined_ds is None:
                combined_ds = ds
                total_steps_per_epoch = steps
                self.is_triplet_dataset = is_triplet
            else:
                combined_ds = combined_ds.concatenate(ds)
                total_steps_per_epoch += steps
                if is_triplet != self.is_triplet_dataset:
                    print(f">>>> [Warning]: Inconsistent dataset types (triplet vs non-triplet) across paths. Using {type}.")
                    self.is_triplet_dataset = is_triplet

            # Check classes for consistency
            label_spec = ds.element_spec[-1]
            if isinstance(label_spec, tuple):
                classes = label_spec[1].shape[-1]
            else:
                classes = label_spec.shape[-1]
            
            if total_classes is None:
                total_classes = classes
            elif total_classes != classes:
                print(f">>>> [Warning]: Inconsistent number of classes ({total_classes} vs {classes}) in path {data_path}. Using {total_classes}.")

        if combined_ds is None:
            print(">>>> [Error]: No valid datasets initialized, combined_ds is None.")
            return

        print(f">>>> Total number of folders/classes across all paths: {total_folders}")  # Added print statement
        self.train_ds, self.steps_per_epoch = combined_ds, total_steps_per_epoch
        self.classes = total_classes

        if tf.distribute.has_strategy():
            self.train_ds = self.train_ds.with_options(self.data_options)

        label_spec = self.train_ds.element_spec[-1]
        if isinstance(label_spec, tuple):
            self.is_distill_ds = True
            self.teacher_emb_size = label_spec[0].shape[-1]
            self.classes = label_spec[1].shape[-1]
            if type == self.distill:
                self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[1:] * len(emb_loss_names) + yy[:1]))
            elif (self.distill in emb_loss_names and len(emb_loss_names) != 1) or (self.distill not in emb_loss_names and len(emb_loss_names) != 0):
                label_data_len = len(emb_loss_names) if self.distill in emb_loss_names else len(emb_loss_names) + 1
                self.train_ds = self.train_ds.map(lambda xx, yy: (xx, yy[:1] + yy[1:] * label_data_len))
        else:
            self.is_distill_ds = False
            self.classes = label_spec.shape[-1]
        print(f"Dataset initialized: classes={self.classes}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
    
    def __init_optimizer__(self, optimizer):
        if optimizer == None:
            if self.model != None and self.model.optimizer != None:
                self.optimizer = self.model.optimizer
                compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
                print(">>>> Reuse optimizer from previous model:", compiled_opt.__class__.__name__)
            else:
                print(">>>> Use default optimizer:", self.default_optimizer)
                self.optimizer = self.default_optimizer
        else:
            print(">>>> Use specified optimizer:", optimizer)
            self.optimizer = optimizer

        try:
            import tensorflow_addons as tfa
        except:
            pass
        else:
            compiled_opt = self.optimizer.inner_optimizer if isinstance(self.optimizer, keras.mixed_precision.LossScaleOptimizer) else self.optimizer
            if isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
                print(">>>> Append weight decay callback...")
                lr_base, wd_base = self.optimizer.lr.numpy(), self.optimizer.weight_decay.numpy()
                wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=self.is_lr_on_batch)
                self.callbacks.append(wd_callback)

    def __init_model__(self, type, loss_top_k=1, header_append_norm=False):
        inputs = self.basic_model.inputs[0]
        embedding = self.basic_model.outputs[0]
        is_multi_output = lambda mm: len(mm.outputs) != 1 or isinstance(mm.layers[-1], keras.layers.Concatenate)
        if self.model != None and is_multi_output(self.model):
            output_layer = min(len(self.basic_model.layers), len(self.model.layers) - 1)
            self.model = keras.models.Model(inputs, self.model.layers[output_layer].output)

        if self.output_weight_decay != 0:
            print(">>>> Add L2 regularizer to model output layer, output_weight_decay = %f" % self.output_weight_decay)
            output_kernel_regularizer = keras.regularizers.L2(self.output_weight_decay / 2)
        else:
            output_kernel_regularizer = None

        model_output_layer_name = None if self.model is None else self.model.output_names[-1]
        if type == self.softmax and model_output_layer_name != self.softmax:
            print(">>>> Add softmax layer...")
            softmax_logits = keras.layers.Dense(self.classes, use_bias=False, name=self.softmax + "_logits", kernel_regularizer=output_kernel_regularizer)
            if self.model != None and "_embedding" not in self.model.output_names[-1]:
                softmax_logits.build(embedding.shape)
                weight_cur = softmax_logits.get_weights()
                weight_pre = self.model.layers[-1].get_weights()
                if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
                    print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
                    softmax_logits.set_weights(weight_pre)
            logits = softmax_logits(embedding)
            output_fp32 = keras.layers.Activation("softmax", dtype="float32", name=self.softmax)(logits)
            self.model = keras.models.Model(inputs, output_fp32)
        elif type == self.arcface and (model_output_layer_name != self.arcface or self.model.layers[-1].append_norm != header_append_norm):
            vpl_start_iters = self.vpl_start_iters * self.steps_per_epoch if self.vpl_start_iters < 50 else self.vpl_start_iters
            vpl_kwargs = {"vpl_lambda": 0.15, "start_iters": vpl_start_iters, "allowed_delta": self.vpl_allowed_delta}
            arc_kwargs = {"loss_top_k": loss_top_k, "append_norm": header_append_norm, "partial_fc_split": self.partial_fc_split, "name": self.arcface}
            print(">>>> Add arcface layer, arc_kwargs={}, vpl_kwargs={}...".format(arc_kwargs, vpl_kwargs))
            if vpl_start_iters > 0:
                batch_size = self.batch_size_per_replica
                arcface_logits = models.NormDenseVPL(batch_size, self.classes, output_kernel_regularizer, **arc_kwargs, **vpl_kwargs, dtype="float32")
            else:
                arcface_logits = models.NormDense(self.classes, output_kernel_regularizer, **arc_kwargs, dtype="float32")

            if self.model != None and "_embedding" not in self.model.output_names[-1]:
                arcface_logits.build(embedding.shape)
                weight_cur = arcface_logits.get_weights()
                weight_pre = self.model.layers[-1].get_weights()
                if len(weight_cur) == len(weight_pre) and weight_cur[0].shape == weight_pre[0].shape:
                    print(">>>> Reload previous %s weight..." % (self.model.output_names[-1]))
                    arcface_logits.set_weights(weight_pre)
            output_fp32 = arcface_logits(embedding)
            self.model = keras.models.Model(inputs, output_fp32)
            # Print the summary of the full model (with ArcFace layer)
            print(">>>> Printing full model summary (with ArcFace layer):")
            self.model.summary()
        elif type in [self.triplet, self.center, self.distill]:
            self.model = self.basic_model
            self.model.output_names[0] = type + "_embedding"
        else:
            print(">>>> Will NOT change model output layer.")

        if self.pretrained is not None:
            if self.model is None:
                self.basic_model.load_weights(self.pretrained)
            else:
                self.model.load_weights(self.pretrained)
            self.pretrained = None
        print(f"Model initialization complete for type: {type}")  # Debug print

    def __add_emb_output_to_model__(self, emb_type, emb_loss, emb_loss_weight):
        nns = self.model.output_names
        emb_shape = self.basic_model.output_shape[-1]
        if emb_type == self.distill and self.teacher_emb_size != emb_shape:
            print(">>>> Add a dense layer to map embedding: student %d --> teacher %d" % (emb_shape, self.teacher_emb_size))
            embedding = self.basic_model.outputs[0]
            if self.distill_emb_map_layer is None:
                self.distill_emb_map_layer = keras.layers.Dense(self.teacher_emb_size, use_bias=False, name="distill_map", dtype="float32")
            emb_map_output = self.distill_emb_map_layer(embedding)
            self.model = keras.models.Model(self.model.inputs[0], [emb_map_output] + self.model.outputs)
        else:
            self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)

        self.model.output_names[0] = emb_type + "_embedding"
        for id, nn in enumerate(nns):
            self.model.output_names[id + 1] = nn
        self.cur_loss, self.loss_weights = [emb_loss, *self.cur_loss], {ii: lossWeight for ii in self.model.output_names}
        self.loss_weights.update({self.model.output_names[0]: emb_loss_weight})

    def __init_type_by_loss__(self, loss):
        print(">>>> Init type by loss function name...")
        if isinstance(loss, str):
            return self.softmax

        if loss.__class__.__name__ == "function":
            ss = loss.__name__.lower()
            if self.softmax in ss:
                return self.softmax
            if self.arcface in ss:
                return self.arcface
            if self.triplet in ss:
                return self.triplet
            if self.distill in ss:
                return self.distill
        else:
            ss = loss.__class__.__name__.lower()
            if isinstance(loss, losses.TripletLossWapper) or self.triplet in ss:
                return self.triplet
            if isinstance(loss, losses.CenterLoss) or self.center in ss:
                return self.center
            if isinstance(loss, losses.ArcfaceLoss) or self.arcface in ss:
                return self.arcface
            if isinstance(loss, losses.ArcfaceLossSimple) or isinstance(loss, losses.AdaCosLoss):
                return self.arcface
            if isinstance(loss, losses.DistillKLDivergenceLoss):
                return self.arcface
            if self.softmax in ss:
                return self.softmax
        return self.softmax

    def __init_emb_losses__(self, embLossTypes=None, embLossWeights=1):
        emb_loss_names, emb_loss_weights = {}, {}
        if embLossTypes is not None:
            embLossTypes = embLossTypes if isinstance(embLossTypes, list) else [embLossTypes]
            for id, ee in enumerate(embLossTypes):
                emb_loss_name = ee.lower() if isinstance(ee, str) else ee.__name__.lower()
                emb_loss_weight = float(embLossWeights[id] if isinstance(embLossWeights, list) else embLossWeights)
                if "centerloss" in emb_loss_name:
                    emb_loss_names[self.center] = losses.CenterLoss if isinstance(ee, str) else ee
                    emb_loss_weights[self.center] = emb_loss_weight
                elif "triplet" in emb_loss_name:
                    emb_loss_names[self.triplet] = losses.BatchHardTripletLoss if isinstance(ee, str) else ee
                    emb_loss_weights[self.triplet] = emb_loss_weight
                elif "distill" in emb_loss_name:
                    emb_loss_names[self.distill] = losses.distiller_loss_cosine if ee == None or isinstance(ee, str) else ee
                    emb_loss_weights[self.distill] = emb_loss_weight
        return emb_loss_names, emb_loss_weights

    def __basic_train__(self, epochs, initial_epoch=0):
        self.model.compile(optimizer=self.optimizer, loss=self.cur_loss, metrics=self.metrics, loss_weights=self.loss_weights)
        print(f"Starting model training: epochs={epochs}, initial_epoch={initial_epoch}, steps_per_epoch={self.steps_per_epoch}")  # Debug print
        self.model.fit(
            self.train_ds,
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks,
            initial_epoch=initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
            use_multiprocessing=True,
            workers=4,
        )
        print("Training completed")  # Debug print

    def reset_dataset(self, data_paths=None):  # Updated to accept data_paths
        self.train_ds = None
        if data_paths != None:
            self.data_paths = data_paths

    def train_single_scheduler(
        self,
        epoch,
        loss=None,
        initial_epoch=0,
        lossWeight=1,
        optimizer=None,
        bottleneckOnly=False,
        lossTopK=1,
        type=None,
        embLossTypes=None,
        embLossWeights=1,
        tripletAlpha=0.35,
    ):
        emb_loss_names, emb_loss_weights = self.__init_emb_losses__(embLossTypes, embLossWeights)

        # If no loss is provided, use a default ArcFace loss
        if loss is None:
            print(">>>> No loss specified, using default ArcFace loss...")
            loss = losses.ArcfaceLoss()
            if self.model is not None and self.model.built:
                print(">>>> Compiling model with default ArcFace loss...")
                self.model.compile(
                    optimizer="adam",
                    loss=loss,
                    metrics=["accuracy"]
                )

        if type is None and not self.inited_from_model:
            type = self.__init_type_by_loss__(loss)
        print(">>>> Train %s..." % type)
        self.__init_dataset__(type, emb_loss_names)
        if self.train_ds is None:
            print(">>>> [Error]: train_ds is None.")
            if self.model is not None:
                self.model.stop_training = True
            return
        if self.is_distill_ds == False and type == self.distill:
            print(">>>> [Error]: Dataset doesn't contain embedding data.")
            if self.model is not None:
                self.model.stop_training = True
            return

        self.is_lr_on_batch = isinstance(self.lr_scheduler, myCallbacks.CosineLrScheduler)
        if self.is_lr_on_batch:
            self.lr_scheduler.steps_per_epoch = self.steps_per_epoch

        basic_callbacks = [ii for ii in [self.my_history, self.model_checkpoint, self.lr_scheduler] if ii is not None]
        self.callbacks = self.my_evals + self.custom_callbacks + basic_callbacks
        self.__init_optimizer__(optimizer)
        if not self.inited_from_model:
            header_append_norm = isinstance(loss, losses.MagFaceLoss) or isinstance(loss, losses.AdaFaceLoss)
            self.__init_model__(type, lossTopK, header_append_norm)

        self.cur_loss, self.loss_weights = [loss], {ii: lossWeight for ii in self.model.output_names}
        if self.center in emb_loss_names and type != self.center:
            loss_class = emb_loss_names[self.center]
            print(">>>> Attach center loss:", loss_class.__name__)
            emb_shape = self.basic_model.output_shape[-1]
            initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
            center_loss = loss_class(self.classes, emb_shape=emb_shape, initial_file=initial_file)
            self.callbacks.append(center_loss.save_centers_callback)
            self.__add_emb_output_to_model__(self.center, center_loss, emb_loss_weights[self.center])

        if self.triplet in emb_loss_names and type != self.triplet:
            loss_class = emb_loss_names[self.triplet]
            print(">>>> Attach triplet loss: %s, alpha = %f..." % (loss_class.__name__, tripletAlpha))
            triplet_loss = loss_class(alpha=tripletAlpha)
            self.__add_emb_output_to_model__(self.triplet, triplet_loss, emb_loss_weights[self.triplet])

        if self.is_distill_ds and type != self.distill:
            distill_loss = emb_loss_names.get(self.distill, losses.distiller_loss_cosine)
            print(">>>> Attach distill loss:", distill_loss.__name__)
            self.__add_emb_output_to_model__(self.distill, distill_loss, emb_loss_weights.get(self.distill, 1))

        print(">>>> loss_weights:", self.loss_weights)
        self.metrics = {ii: None if "embedding" in ii else "accuracy" for ii in self.model.output_names}
        self.callbacks.append(myCallbacks.ExitOnNaN())
        if self.vpl_start_iters > 0:
            loss.build(self.batch_size_per_replica)
            self.callbacks.append(myCallbacks.VPLUpdateQueue())

        if self.gently_stop:
            self.callbacks.append(self.gently_stop)

        if bottleneckOnly:
            print(">>>> Train bottleneckOnly...")
            self.basic_model.trainable = False
            self.callbacks = self.callbacks[len(self.my_evals) :]
            self.__basic_train__(epoch, initial_epoch=0)
            self.basic_model.trainable = True
        else:
            self.__basic_train__(initial_epoch + epoch, initial_epoch=initial_epoch)

        print(">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s" % (type, self.model.history.epoch, self.model.stop_training))
        print(">>>> My history:")
        self.my_history.print_hist()
        latest_save_path = os.path.join("checkpoints", os.path.splitext(self.save_path)[0] + "_basic_model_latest.h5")
        print(">>>> Saving latest basic model to:", latest_save_path)
        self.basic_model.save(latest_save_path)

    def train(self, train_schedule, initial_epoch=0):
        train_schedule = [train_schedule] if isinstance(train_schedule, dict) else train_schedule
        for sch in train_schedule:
            for ii in ["centerloss", "triplet", "distill"]:
                if ii in sch:
                    sch.setdefault("embLossTypes", []).append(ii)
                    sch.setdefault("embLossWeights", []).append(sch.pop(ii))
            if "alpha" in sch:
                sch["tripletAlpha"] = sch.pop("alpha")

            self.train_single_scheduler(**sch, initial_epoch=initial_epoch)
            initial_epoch += 0 if sch.get("bottleneckOnly", False) else sch["epoch"]

            if self.model is None or self.model.stop_training == True:
                print(">>>> But it's an early stop, break...")
                break
        return initial_epoch

if __name__ == "__main__":
    print("Entering main execution block")  # Debug print
    # Prompt user for multiple dataset directories at runtime
    data_paths = []
    while True:
        data_path = input("Please enter a dataset directory path (or press Enter to finish): ").strip()
        if data_path == "":
            break
        data_paths.append(data_path)
    
    if not data_paths:
        print(">>>> [Error]: No dataset paths provided. Exiting.")
        sys.exit(1)
    
    print(f"Dataset directories provided: {data_paths}")  # Debug print
    trainer = Train(
        data_paths=data_paths,  # Pass list of paths
        save_path="train",
        eval_paths=[],
        basic_model=None,
        model=None,
        compile=True,
        output_weight_decay=1,
        custom_objects={},
        pretrained=None,
        batch_size=8,
        lr_base=0.001,
        lr_decay=0.05,
        lr_decay_steps=0,
        lr_min=1e-6,
        lr_warmup_steps=0,
        eval_freq=1,
        random_status=0,
        random_cutout_mask_area=0.0,
        image_per_class=0,
        samples_per_mining=0,
        mixup_alpha=0,
        partial_fc_split=0,
        teacher_model_interf=None,
        sam_rho=0,
        vpl_start_iters=-1,
        vpl_allowed_delta=200,
    )
    print("Train instance created")  # Debug print
    trainer.train_single_scheduler(
        epoch=10,
        loss=None,
        initial_epoch=0,
        lossWeight=1,
        optimizer=None,
        bottleneckOnly=False,
        lossTopK=1,
        type=None,
        embLossTypes=None,
        embLossWeights=1,
        tripletAlpha=0.35
    )