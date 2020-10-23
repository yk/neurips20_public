import sys
import shutil
import copy
import logging
import tensorflow as tf
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np
import itertools as itt
import functools as fct
import tensorflow.contrib.metrics as tfm
import os.path


def tf_print(a):
    return tf.Print(a, [tf.reduce_mean(a)])


def variables_by_prefix(prefix):
    vs = tf.global_variables()
    return [v for v in vs if v.name.startswith(prefix)]


def variable_by_name_end(name):
    vs = tf.get_collection('variables')
    for v in vs:
        if v.name.endswith(name):
            return v
    for v in vs:
        print(v.name)
    raise Exception('Variable {} not found'.format(name))

def variable_by_name(name):
    vs = tf.get_collection('variables')
    for v in vs:
        if v.name == name:
            return v
    for v in vs:
        print(v.name)
    raise Exception('Variable {} not found'.format(name))


class Callback:
    def __init__(self, steps_per_epoch=1):
        self.steps_per_epoch = steps_per_epoch

    def setup(self, trainer):
        self.trainer = trainer
        self._setup()

    def setup_after_trainer(self):
        self._setup_after_trainer()

    def _setup(self):
        pass

    def _setup_after_trainer(self):
        pass

    def before_init(self):
        self._before_init()

    def _before_init(self):
        pass

    def after_init(self):
        self._after_init()

    def _after_init(self):
        pass

    def is_epoch_now(self):
        if self.steps_per_epoch <= 0:
            return False
        return self.trainer.step_count % self.steps_per_epoch == 0

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        pass

    def trigger_step(self):
        self._trigger_step()
        if self.is_epoch_now():
            self.trigger_epoch()

    def _trigger_step(self):
        pass


class ArgvPrinter(Callback):
    def _trigger_epoch(self):
        logging.info(" ".join(sys.argv))


class ModelSaver(Callback):
    def __init__(self, *args, write_meta_graph=True, max_to_keep=None, var_prefix=None, save_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_meta_graph = write_meta_graph
        self.max_to_keep = max_to_keep
        self.prefix = var_prefix
        self.save_dir = save_dir

    def _setup(self):
        self._to_save = tf.global_variables()
        if self.prefix:
            self._to_save = [v for v in self._to_save if v.name.startswith(self.prefix)]
        self.saver = tf.train.Saver(self._to_save, max_to_keep=self.max_to_keep)

    def _trigger_epoch(self):
        logging.info('Saving model')
        path = './logs/model.ckpt'
        if self.save_dir is not None:
            path = './logs/{}/model.ckpt'.format(self.save_dir)
        self.saver.save(self.trainer.sess, path, write_meta_graph=self.write_meta_graph, global_step=self.trainer.step_count)


class ModelRestorer(Callback):
    def __init__(self, path, var_prefix=None):
        super().__init__()
        self.path = path
        self.prefix = var_prefix

    def _setup(self):
        self._to_restore = tf.global_variables()
        if self.prefix:
            self._to_restore = [v for v in self._to_restore if v.name.startswith(self.prefix)]
        self.saver = tf.train.Saver(self._to_restore)

    def _before_init(self):
        self.saver.restore(self.trainer.sess, self.path)
        self.trainer.init_op = tf.group(
                tf.local_variables_initializer(), 
                tf.variables_initializer([v for v in tf.global_variables() if v not in self._to_restore])
            )


class EpochCounter(Callback):
    def __init__(self, epochs, total_epochs=None):
        super().__init__(epochs)
        self.total_epochs = total_epochs

    def _setup(self):
        self.epochs = 0

    def _trigger_epoch(self):
        logging.info('Epoch: {}'.format(self.epochs) + ('' if self.total_epochs is None else ' of {}'.format(self.total_epochs)))
        self.epochs += 1


class StepDisplay(Callback):
    def _create_pbar(self):
        self.pbar = tqdm(total=self.steps_per_epoch)

    def _trigger_step(self):
        if self.is_epoch_now():
            if hasattr(self, 'pbar'):
                self.pbar.close()
            self._create_pbar()
        self.pbar.update(1)


class Evaluator:
    def setup(self, runner):
        self.runner = runner
        self._setup()

    def _setup(self):
        pass

    def setup_after_trainer(self):
        self._setup_after_trainer()

    def _setup_after_trainer(self):
        pass

    def after_init(self):
        self._after_init()

    def _after_init(self):
        pass

    def _before_epoch_ops(self):
        return []

    def _get_ops(self):
        return []

    def trigger_epoch(self, summary):
        self._trigger_epoch(summary)

    def _trigger_epoch(self, summary):
        pass


def streaming_mean(name, value):
    tf.summary.scalar(name, tfm.streaming_mean(value, name='stream/{}'.format(name))[1])


def streaming_accuracy(name, predictions, labels):
    tf.summary.scalar(name, tfm.streaming_accuracy(predictions, labels, name='stream/{}'.format(name))[1])

def streaming_concat(name, value, axis=0):
    tf.summary.scalar(name, tf.reduce_mean(tfm.streaming_concat(value, axis=axis, name='stream/{}'.format(name))[1]))


class EvalDatasetRunner(Callback):
    def __init__(self, steps_per_epoch, model, create_queue_or_iter, evaluators=[], eval_steps=1):
        super().__init__(steps_per_epoch)
        self.create_queue_or_iter = create_queue_or_iter
        self.model = model
        self.evaluators = evaluators
        self.eval_steps = eval_steps

    def _setup(self):
        self.data = self.create_queue_or_iter()

        with tf.variable_scope('', reuse=True), no_training_context():
            self.model.build_graph(self.data)

        before_epoch_ops = []
        eval_ops = []
        for e in self.evaluators:
            e.setup(self)
            beops = e._before_epoch_ops()
            if beops is not None:
                before_epoch_ops += beops
            eops = e._get_ops()
            if eops is not None:
                eval_ops += eops
        stream_vars = [v for v in tf.local_variables() if 'stream/' in v.name]
        logging.info('---STREAM VARS---')
        for v in stream_vars:
            logging.info(v.name)

        self.before_epoch_op = tf.group(tf.no_op(), *before_epoch_ops)

        self.stream_reset_op = tf.variables_initializer(stream_vars)
        with tf.control_dependencies(eval_ops):
            self.summary_op = tf.identity(tf.summary.merge_all(tf.GraphKeys.SUMMARIES))

    def _setup_after_trainer(self):
        for e in self.evaluators:
            e.setup_after_trainer()

    def _after_init(self):
        for e in self.evaluators:
            e.after_init()

    def _trigger_epoch(self):
        if self.trainer.step_count == 0:
            return
        self.trainer.sess.run(self.stream_reset_op)
        self.trainer.sess.run(self.before_epoch_op)
        for _ in tqdm(range(self.eval_steps), desc="EVAL"):
            if self.model.feed:
                batch = next(self.data)
            else:
                batch = None
            feed = self.model.build_feed_dict(batch)
            self.current_batch = batch
            self.current_feed = feed
            summary_str = self.trainer.sess.run(self.summary_op, feed_dict=feed)
        self.trainer.summary_writer.add_summary(summary_str, self.trainer.step_count + 1)
        self.trainer.summary_writer.flush()

        summ = tf.Summary()
        summ.ParseFromString(summary_str)
        for val in summ.value:
            if val.HasField('simple_value'):
                logging.info('{}: {}'.format(val.tag, val.simple_value))

        for e in self.evaluators:
            e.trigger_epoch(summ)

TRAINING_SUMMARY_KEY = 'training_summaries'


class Trainer:
    def __init__(self, model, create_queue_or_iter, callbacks=[], write_train_summaries=True, train_data_size=None, max_steps=None, train_log_steps=1):
        self.model = model
        self.create_queue_or_iter = create_queue_or_iter
        self.callbacks = callbacks
        self.write_train_summaries = write_train_summaries
        self.train_data_size = train_data_size
        self.max_steps = max_steps
        self.train_log_steps = train_log_steps

    def setup(self):
        self.data = self.create_queue_or_iter()

        with training_context():
            self.model.build_graph(self.data)
        self._setup_callbacks()
        self._setup()
        self._setup_callbacks_after_trainer()
        shutil.rmtree('./logs', ignore_errors=True)
        self.summary_writer = tf.summary.FileWriter('./logs')
        count_params()
        self.sess = tf.Session()
        self._before_init()
        print('----TRAINABLES----')
        for v in tf.trainable_variables():
            print(v)
        tf.get_default_graph().finalize()
        self.sess.run(self.init_op)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        self._after_init()

    def _setup_callbacks(self):
        for c in self.callbacks:
            c.setup(self)

    def _setup_callbacks_after_trainer(self):
        for c in self.callbacks:
            c.setup_after_trainer()

    def _before_init(self):
        for c in self.callbacks:
            c.before_init()

    def _after_init(self):
        for c in self.callbacks:
            c.after_init()

    def build_train_op(self, ops, global_step_op=None, training_summary_op=None, do_step=True):
        if global_step_op is None:
            gsv = get_global_step_var()
            if do_step:
                global_step_op = tf.assign_add(gsv, 1)
            else:
                global_step_op = tf.assign_add(gsv, 0)
        if training_summary_op is None:
            if do_step and self.write_train_summaries:
                training_summary_op = tf.summary.merge_all(TRAINING_SUMMARY_KEY)
            else:
                training_summary_op = tf.constant('', dtype=tf.string)
        train_op = tf.tuple((training_summary_op, global_step_op), control_inputs=[ops], name='train_op')
        return train_op

    def _get_train_ops(self):
        return [] 

    def _setup(self):
        tos = self._get_train_ops()
        main_to = self.build_train_op(tos[0], do_step=True)
        other_tos = [self.build_train_op(to, do_step=True) for to in tos[1:]]
        self.train_ops = itt.cycle([main_to] + other_tos)
        self.num_train_ops = len(other_tos) + 1
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def run_step(self):
        return self._run_step()

    def _run_step(self):
        if self.model.feed:
            batch = next(self.data)
        else:
            batch = None
        feed = self.model.build_feed_dict(batch)
        summary_str, global_step = self.sess.run(next(self.train_ops), feed_dict=feed)
        did_step = global_step == self.step_count + 1
        if self.write_train_summaries and len(summary_str) > 0 and did_step and global_step % self.train_log_steps == 0:
            self.summary_writer.add_summary(summary_str, global_step=global_step)
            self.summary_writer.flush()
        return did_step

    def _stop(self):
        pass

    def stop(self):
        if self.max_steps:
            if self.step_count >= self.max_steps:
                return True
        return self._stop() or self.coord.should_stop()

    def run_callbacks(self):
        for c in self.callbacks:
            c.trigger_step()

    def train(self):
        with tf.Graph().as_default():
            self.step_count = 0
            self.setup()
            stop_it = False
            try:
                while not self.stop() and not stop_it:
                    try:
                        did_step = self.run_step()
                    except (StopIteration, tf.errors.OutOfRangeError):
                        stop_it = True
                        did_step = False
                    self.run_callbacks()
                    if did_step:
                        self.step_count += 1
            finally:
                self.coord.request_stop()
            self.coord.join(self.threads)
            self.sess.close()


class ModelDesc:
    @property
    def feed(self):
        return self.get_input_vars() is not None

    def build_graph(self, input_vars=None):
        if self.feed:
            input_vars = self.get_input_vars()
        return self._build_graph(input_vars)

    def _build_graph(self, input_vars):
        pass

    def build_feed_dict(self, batch=None):
        if not self.feed:
            return None
        return dict(zip(self.get_input_vars(), batch))

    def get_input_vars(self):
        if not hasattr(self.__class__, '_input_vars'):
            self.__class__._input_vars = self._get_input_vars()
        return self.__class__._input_vars

    def _get_input_vars(self):
        pass


GLOBAL_STEP_VAR_NAME = 'global_step_var:0'
GLOBAL_STEP_OP_NAME = 'global_step_var'


def get_global_step_var():
    """
    Returns:
        tf.Tensor: the global_step variable in the current graph. create if
        doesn't exist.
    """
    try:
        return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    except KeyError:
        scope = tf.get_variable_scope()
        assert scope.name == '', \
            "Creating global_step_var under a variable scope would cause problems!"
        with tf.variable_scope(scope, reuse=False):
            var = tf.get_variable(GLOBAL_STEP_OP_NAME, shape=[],
                                  initializer=tf.constant_initializer(dtype=tf.int32),
                                  trainable=False, dtype=tf.int32)
        return var


def get_global_step(sess):
    """
    Returns:
        float: global_step value in current graph and session"""
    # return tf.train.global_step(
        # tf.get_default_session(),
        # get_global_step_var())
    return get_global_step_var().eval(session=sess)


_ypack_global_context = None


def get_global_context():
    global _ypack_global_context
    if _ypack_global_context is None:
        _ypack_global_context = dict()
    return _ypack_global_context


@contextmanager
def _context(**kwargs):
    gc = get_global_context()
    c = copy.deepcopy(gc)
    gc.update(kwargs)
    yield
    gc.clear()
    gc.update(c)
    return


@contextmanager
def training_context(**kwargs):
    kwargs['training'] = True
    c = _context().func(**kwargs)
    next(c)
    yield
    next(c)
    return


@contextmanager
def no_training_context(**kwargs):
    kwargs['training'] = False
    c = _context().func(**kwargs)
    next(c)
    yield
    next(c)
    return


def context_var(key, default=None):
    return get_global_context().get(key, default)


def set_default_context(**kwargs):
    global _ypack_global_context
    if _ypack_global_context is not None:
        logging.warn("default context set twice")
    _ypack_global_context = kwargs


def is_training():
    return context_var('training', False)


def random_select(n, k, resample=False):
    if resample:
        sel = tf.random_uniform([k], maxval=n, dtype=tf.int32)
    else:
        r = np.arange(n, dtype=np.int32)
        s = tf.random_shuffle(r)
        sel = s[:k]
    return sel


def lrelu(x, alpha=0.2):
    xtop = tf.nn.relu(x)
    xbot = tf.nn.relu(-xtop)
    xx = xtop - tf.constant(alpha, dtype=tf.float32) * xbot
    return xx


#http://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
def count_params():
    "print number of trainable variables"
    size = lambda v: fct.reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print("Model size: %dK" % (n//1000,))



from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.eager import context

class OptimisticRestoreSaver(Saver):
    """Only restores variables in `var_list` that are present in the checkpoint on restore. However, on save, all variables in `var_list` are written to the checkpoint."""
    def restore(self, sess, save_path, var_filter=lambda v: True):
        """Restores only variables that are contained in `save_path` and match in shape and dtype and return `True` when passed to `var_filter`."""
        if self._is_empty:
            return
        if save_path is None:
            raise ValueError("Can't load save_path when it is None.")
        logging.info("Restoring parameters from %s", save_path)

        reader = tf.train.load_checkpoint(save_path)
        shape_map = reader.get_variable_to_shape_map()
        dtype_map = reader.get_variable_to_dtype_map()

        restore_op_name = self.saver_def.restore_op_name
        restore_op_grouped = sess.graph.get_operation_by_name(restore_op_name)

        restore_ops = []
        for r_op in restore_op_grouped.control_inputs:
            v = r_op.inputs[0]
            tensor_name = v.op.name
            tensor_shape = v.get_shape().as_list()
            tensor_dtype = v.dtype.base_dtype
            if tensor_name not in shape_map or tensor_name not in dtype_map:
                logging.warn('variable %s not in checkpoint', tensor_name)
            elif shape_map[tensor_name] != tensor_shape:
                logging.warn('variable %s in checkpoint, but checkpoint shape %r does not match graph shape %r', tensor_name, shape_map[tensor_name], tensor_shape)
            elif dtype_map[tensor_name] != tensor_dtype:
                logging.warn('variable %s in checkpoint, but checkpoint dtype %r does not match graph dtype %r', tensor_name, dtype_map[tensor_name], tensor_dtype)
            elif not var_filter(v):
                logging.info('variable %s rejected by var_filter', tensor_name, dtype_map[tensor_name], tensor_dtype)
            else:
                restore_ops.append(r_op)
                logging.info('adding variable %s to be restored', tensor_name)

        if context.in_graph_mode():
            for r_op in restore_ops:
                sess.run(r_op,
                    {self.saver_def.filename_tensor_name: save_path})
        else:
            raise NotImplementedError("eager selective restoring not supprted yet")

