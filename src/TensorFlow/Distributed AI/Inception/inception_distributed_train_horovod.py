'''
Created on Jul 10, 2018

@author: MYue
'''
"""A library to train Inception using multiple replicas with synchronous update.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data.build_image_data import FLAGS

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import image_processing
import inception_model as inception
from slim import slim1

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Communication protocol to use in distributed """
                           """execution (default grpc) """)

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in SyncReplicasOptimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the SyncReplicasOptimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# from tensorflow.python.client import device_lib
# Local_device_protos = device_lib.list_local_devices()
# print([x.name for x in Local_device_protos if x.device_type=='GPU'])


def train(target, dataset, cluster_spec):
  """Train Inception on a dataset for a number of steps."""
  # Number of workers and parameter servers are inferred from the workers and ps
  # hosts string.
  
  hvd.init()

  # Choose worker 0 as the chief. Note that any worker could be the chief
  # but there should be only one chief.
  FLAGS.task_id = hvd.rank()
  is_chief = (FLAGS.task_id == 0)
  print('horovod rank: %d ' % hvd.rank())

    # Variables and its related init/assign ops are assigned to ps.
  with slim1.scopes.arg_scope(
        [slim1.variables.variable, slim1.variables.global_step],
        device=slim1.variables.VariableDeviceChooser(#num_parameter_servers
                                                     0 )):
      # Create a variable to count the number of train() calls. This equals the
      # number of updates applied to the variables.
      global_step = slim1.variables.global_step()

      # Calculate the learning rate schedule.
      num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                               FLAGS.batch_size)
      # Decay steps need to be divided by the number of replicas to aggregate.
      decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay /
                        #num_replicas_to_aggregate
                        hvd.size())

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)
      # Add a summary to track the learning rate.
      tf.summary.scalar('learning_rate', lr)

      # Create an optimizer that performs gradient descent.
      opt = tf.train.RMSPropOptimizer(lr * hvd.size(),
                                      RMSPROP_DECAY,
                                      momentum=RMSPROP_MOMENTUM,
                                      epsilon=RMSPROP_EPSILON)

      images, labels = image_processing.distorted_inputs(
          dataset,
          batch_size=FLAGS.batch_size,
          num_preprocess_threads=FLAGS.num_preprocess_threads)

      # Number of classes in the Dataset label set plus 1.
      # Label 0 is reserved for an (unused) background class.
      num_classes = dataset.num_classes() + 1
      logits = inception.inference(images, num_classes, for_training=True)
      # Add classification loss.
      inception.loss(logits, labels)

      # Gather all of the losses including regularization losses.
      losses = tf.get_collection(slim1.losses.LOSSES_COLLECTION)
      losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

      total_loss = tf.add_n(losses, name='total_loss')


        # Compute the moving average of all individual losses and the
        # total loss.
      loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
      loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summmary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
      for l in losses + [total_loss]:
          loss_name = l.op.name
          # Name each loss as '(raw)' and name the moving average version of the
          # loss as the original loss name.
          tf.summary.scalar(loss_name + '_raw_', l)
          tf.summary.scalar(loss_name, loss_averages.average(l))

        # Add dependency to compute loss_averages.
      with tf.control_dependencies([loss_averages_op]):
          total_loss = tf.identity(total_loss)

      # Track the moving averages of all trainable variables.
      # Note that we maintain a 'double-average' of the BatchNormalization
      # global statistics.
      # This is not needed when the number of replicas are small but important
      # for synchronous distributed training with tens of workers/replicas.

      
      batchnorm_updates = tf.get_collection(slim1.ops.UPDATE_OPS_COLLECTION)
      assert batchnorm_updates, 'Batchnorm updates are missing'
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      # Add dependency to compute batchnorm_updates.
      with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(total_loss)

      # Compute gradients with respect to the loss.
      grads = opt.compute_gradients(total_loss)

      # Add histograms for gradients.
      for grad, var in grads:
        if grad is not None:
          tf.summary.histogram(var.op.name + '/gradients', grad)

      apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
       
      # Horovod: add Horovod Distributed Optimizer.
      opt = hvd.DistributedOptimizer(opt)


      with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.identity(total_loss, name='train_op')

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.summary.merge_all()
      
      summary_hook = tf.train.SummarySaverHook(save_steps=100,output_dir=FLAGS.train_dir,summary_op=summary_op) 
      
      hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),
        
        summary_hook,

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=20000 // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': total_loss},
                                   every_n_iter=10)
      ]

      # Build an initialization operation to run below.
      init_op = tf.global_variables_initializer()

      # We run the summaries in the same thread as the training operations by
      # passing in None for summary_op to avoid a summary_thread being started.
      # Running summaries and training operations in parallel could run out of
      # GPU memory.
      
      sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement,
          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth = True,
                                    visible_device_list = str(hvd.local_rank())))
      
      checkpoint_dir = FLAGS.train_dir if hvd.rank() == 0 else None
      
      with tf.train.MonitoredTrainingSession(
#                             master=target,
#                             is_chief=is_chief,
                            checkpoint_dir= checkpoint_dir,  #FLAGS.train_dir,
                            hooks=hooks,    #[sync_replicas_hook,summary_hook],
                            config=sess_config
                            ) as sess:
          
          # Start the queue runners.
          queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)

          tf.train.start_queue_runners(sess)
          tf.logging.info('Started %d queues for processing input data.',
                          len(queue_runners))
    
    
          # Train, checking for Nans. Concurrently run the summary operation at a
          # specified interval. Note that the summary_op and train_op never run
          # simultaneously in order to prevent running out of GPU memory.
          next_summary_time = time.time() + FLAGS.save_summaries_secs
          
          
          step_value = 0
          
          while not sess.should_stop():
            try:
              start_time = time.time()
              loss_value, step = sess.run([train_op, global_step])
              assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
              if step > FLAGS.max_steps:
                break
              duration = time.time() - start_time
    
              if step % 30 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('Worker %d: %s: step %d, loss = %.2f'
                              '(%.1f examples/sec; %.3f  sec/batch)')
                tf.logging.info(format_str %
                                (FLAGS.task_id, datetime.now(), step, loss_value,
                                 examples_per_sec, duration))
    
            except:
              if is_chief:
                tf.logging.info('Chief got exception while running!')
              raise
    
          # Stop the supervisor.  This also waits for service threads to finish.
          sess.stop()
          
  