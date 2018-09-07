"""Evaluate GRA."""
import os
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import Series, DataFrame

from model.gra import GRA
import config as cfg
import util.dataset as dt

lajidaima = 0


def train(epoch, dataset, config, log_dir):
    """Train model for one epoch."""
    model_config = config['model']
    train_config = config['train']
    sess_config = config['session']

    with tf.Graph().as_default():
        model = GRA(model_config)
        model.build_inference()
        model.build_loss(train_config['reg_coeff'], train_config['shu_coeff'])
        model.build_train(train_config['learning_rate'])
        lajidaima = 0
        with tf.Session(config=sess_config) as sess:
            sum_dir = os.path.join(log_dir, 'summary')
            # create event file for graph
            if not os.path.exists(sum_dir):
                summary_writer = tf.summary.FileWriter(sum_dir, sess.graph)
                summary_writer.close()
            summary_writer = tf.summary.FileWriter(sum_dir)

            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if ckpt_path:
                print('load checkpoint {}.'.format(ckpt_path))
                lajidaima = int(ckpt_path.split('-')[-1]) - epoch + 1
                saver.restore(sess, ckpt_path)
            else:
                print('no checkpoint.')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                sess.run(tf.global_variables_initializer())
            epoch += lajidaima
            stats_dir = os.path.join(log_dir, 'stats')
            stats_path = os.path.join(stats_dir, 'train.json')
            if os.path.exists(stats_path):
                print('load stats file {}.'.format(stats_path))
                stats = pd.read_json(stats_path, 'records')
            else:
                print('no stats file.')
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)
                stats = pd.DataFrame(columns=['epoch', 'loss', 'acc'])

            # train iterate over batch
            batch_idx = 0
            total_loss = 0
            total_acc = 0
            batch_total = np.sum(dataset.train_batch_total)

            while dataset.has_train_batch:
                vgg, c3d, question, answer = dataset.get_train_batch()
                c3d = np.zeros((len(c3d), len(c3d[0]), len(c3d[0][0])))
                feed_dict = {
                    model.appear: vgg,
                    model.motion: c3d,
                    model.question_encode: question,
                    model.answer_encode: answer
                }
                _, loss, prediction = sess.run(
                    [model.train, model.loss, model.prediction], feed_dict)

                # cal acc
                correct = 0
                for i, row in enumerate(prediction[1]):
                    #print(row)
                    for index in row:
                        if answer[i][index] == 1:
                            correct += 1
                            break
                acc = correct / len(answer)

                total_loss += loss
                total_acc += acc
                if batch_idx % 10 == 0:
                    print('[TRAIN] epoch {}, batch {}/{}, loss {:.5f}, acc {:.5f}.'.format(
                        epoch, batch_idx, batch_total, loss, acc))
                batch_idx += 1

            loss = total_loss / batch_total
            acc = total_acc / batch_total
            print('\n[TRAIN] epoch {}, loss {:.5f}, acc {:.5f}.\n'.format(
                epoch, loss, acc))

            summary = tf.Summary()
            summary.value.add(tag='train/loss', simple_value=float(loss))
            summary.value.add(tag='train/acc', simple_value=float(acc))
            summary_writer.add_summary(summary, epoch)

            record = Series([epoch, loss, acc], ['epoch', 'loss', 'acc'])
            stats = stats.append(record, ignore_index=True)

            saver.save(sess, os.path.join(ckpt_dir, 'model.ckpt'), epoch)
            stats.to_json(stats_path, 'records')
            dataset.reset_train()
            return loss, acc


def val(epoch, dataset, config, log_dir):
    """Validate model."""
    model_config = config['model']
    sess_config = config['session']

    answerset = pd.read_csv(
        os.path.join(config['preprocess_dir'], 'answer_set.txt'), header=None)[0]

    example_id = 0

    with tf.Graph().as_default():
        model = GRA(model_config)
        model.build_inference()
        result = DataFrame(columns=['id', 'answer'])
        with tf.Session(config=sess_config) as sess:
            sum_dir = os.path.join(log_dir, 'summary')
            summary_writer = tf.summary.FileWriter(sum_dir)

            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if save_path:
                print('load checkpoint {}.'.format(save_path))
                saver.restore(sess, save_path)
            else:
                print('no checkpoint.')
                exit()

            stats_dir = os.path.join(log_dir, 'stats')
            stats_path = os.path.join(stats_dir, 'val.json')
            if os.path.exists(stats_path):
                print('load stats file {}.'.format(stats_path))
                stats = pd.read_json(stats_path, 'records')
            else:
                print('no stats file.')
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)
                stats = pd.DataFrame(columns=['epoch', 'acc'])

            # val iterate over examples
            correct = 0

            while dataset.has_val_example:
                vgg, c3d, question, answer = dataset.get_val_example()
                c3d = np.zeros((len(c3d), len(c3d[0])))
                feed_dict = {
                    model.appear: [vgg],
                    model.motion: [c3d],
                    model.question_encode: [question],
                }
                prediction = sess.run(model.prediction, feed_dict=feed_dict)
                prediction = prediction[1]
                for i, row in enumerate(prediction):
                    for index in row:
                        if answer[index] == 1:
                            correct += 1
                            break
                result = result.append(
                    {'id': example_id, 'answer': prediction}, ignore_index=True)
                example_id += 1
            acc = correct / dataset.val_example_total
            result.to_json(os.path.join(
                log_dir, 'validation_' + str(int(acc * 100)) + '_' + str(epoch + lajidaima) + '.json'), 'records')
            print('\n[VAL] epoch {}, acc {:.5f}.\n'.format(
                epoch + lajidaima, acc))

            summary = tf.Summary()
            summary.value.add(tag='val/acc', simple_value=float(acc))
            summary_writer.add_summary(summary, epoch + lajidaima)

            record = Series([epoch + lajidaima, acc], ['epoch', 'acc'])
            stats = stats.append(record, ignore_index=True)
            stats.to_json(stats_path, 'records')

            dataset.reset_val()
            return acc


def test(dataset, config, log_dir):
    """Test model, output prediction as json file."""
    model_config = config['model']
    sess_config = config['session']

    answerset = pd.read_csv(
        os.path.join(config['preprocess_dir'], 'answer_set.txt'), header=None)[0]

    with tf.Graph().as_default():
        model = GRA(model_config)
        model.build_inference()

        with tf.Session(config=sess_config) as sess:
            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if save_path:
                print('load checkpoint {}.'.format(save_path))
                saver.restore(sess, save_path)
            else:
                print('no checkpoint.')
                exit()

            # test iterate over examples
            result = DataFrame(columns=['id', 'answer'])
            correct = 0

            while dataset.has_test_example:
                vgg, c3d, question, answer, example_id = dataset.get_test_example()
                feed_dict = {
                    model.appear: [vgg],
                    model.motion: [c3d],
                    model.question_encode: [question],
                }
                prediction,  channel_weight, appear_weight, motion_weight = sess.run(
                    [model.prediction, model.channel_weight, model.appear_weight, model.motion_weight], feed_dict=feed_dict)
                #prediction = prediction[0]
                channel_weight = channel_weight[0]
                appear_weight = appear_weight[0]
                motion_weight = motion_weight[0]

                result = result.append(
                    {'id': example_id, 'answer': prediction[1]}, ignore_index=True)
                # modified-why
                # if answerset[prediction] in answer:
                #     correct += 1
                #     print(answer, example_id, channel_weight)
                # print(appear_weight)
                # print(motion_weight)

            result.to_json(os.path.join(
                log_dir, 'prediction.json'), 'records')

            # acc = correct / dataset.test_example_total
            # print('\n[TEST] acc {:.5f}.\n'.format(acc))

            dataset.reset_test()
            return None


def main():
    """Main script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test',
                        help='train/test')
    parser.add_argument('--gpu', default='0',
                        help='gpu id')
    parser.add_argument('--log', default='./log_gra',
                        help='log directory')
    parser.add_argument('--dataset', choices=['msvd_qa', 'msrvtt_qa'], default='msrvtt_qa',
                        help='dataset name, msvd_qa/msrvtt_qa')
    parser.add_argument('--config', default='0',
                        help='config id')
    args = parser.parse_args()

    config = cfg.get('gra', args.dataset, args.config, args.gpu)

    if args.dataset == 'msvd_qa':
        dataset = dt.MSVDQA(
            config['train']['batch_size'], config['preprocess_dir'])
    elif args.dataset == 'msrvtt_qa':
        dataset = dt.MSRVTTQA(
            config['train']['batch_size'], config['preprocess_dir'], config['model']['answer_num'])

    if args.mode == 'train':
        best_val_acc = -1
        val_acc = 0
        not_improved = -1

        for epoch in range(0, 3000):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                not_improved = 0
            else:
                not_improved += 1
            if not_improved == 10:
                print('early stopping.')
                break

            train(epoch, dataset, config, args.log)
            val_acc = val(epoch, dataset, config, args.log)

    elif args.mode == 'test':
        print('start test.')
        test(dataset, config, args.log)


if __name__ == '__main__':
    main()
