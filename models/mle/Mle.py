from time import time

from models.Gan import Gan
from models.mle.MleDataLoader import DataLoader
from models.mle.MleGenerator import Generator
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleLstm import OracleLstm
from utils.utils import *

def my_generate_samples(sess, trainable_model, batch_size, sequence_length, generate_num, output_file=None, get_code=False, 
                        g_x_file=None, g_grade_file=None, g_class_file=None):
    # Generate Samples
    generated_samples = []
    data_loader = DataLoader(batch_size=batch_size, seq_length=sequence_length)
    grade_data_loader = DataLoader(batch_size=batch_size, seq_length=sequence_length)
    class_data_loader = DataLoader(batch_size=batch_size, seq_length=sequence_length)
    #data_loader.create_batches(data_file=g_x_file)
    grade_data_loader.create_batches(data_file=g_grade_file)
    class_data_loader.create_batches(data_file=g_class_file)

    for _ in range(generate_num // batch_size):
        #g_x = data_loader.next_batch()
        g_grade = grade_data_loader.next_batch()
        g_class = class_data_loader.next_batch()
        #g_x = g_x.reshape((batch_size, sequence_length)).astype(np.int32)
        #g_x = np.zeros((batch_size, sequence_length))
        generated_samples.extend(trainable_model.generate(sess, g_grade, g_class))
    codes = []
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                for item in poem:
                    buffer = (' '.join([str(x) for x in item]) + '\n').replace('[', '').replace(']', '')
                    fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    # codes = ""
    # for poem in generated_samples:
    #     tmp = []
    #     for item in poem:
    #         tmp.extend(item)
    #     buffer = ' '.join([str(x) for x in item]) + '\n'.replace('[', '').replace(']', '')
    #     codes += buffer
    # return codes

class Mle(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 14994
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 2048
        self.start_token = 0
        
        self.add_data = True # add_data can't = False
        save_path = 'models/mle/save/'
        data_path = 'data/task_data/'
        self.oracle_file = save_path + 'oracle.txt'
        self.generator_file = save_path + 'generator.txt'
        self.test_file = save_path + 'test_file.txt'
        self.g_x_file = data_path + 'g_x_file.txt'
        self.grade_file = data_path + 'qnyh_task_data_grades_50k.txt'
        self.class_file = data_path + 'qnyh_task_data_classes_50k.txt'
        self.g_grade_file = data_path + 'g_grades.txt'
        self.g_class_file = data_path + 'g_classes.txt'

    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = None

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        #from utils.metrics.DocEmbSim import DocEmbSim
        #docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        #self.add_metric(docsim)

    def train_discriminator(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            _ = self.sess.run(self.discriminator.train_op, feed)

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score)+',')
            self.log.write('\n')
            return scores
        return super().evaluate()

    def train_oracle(self):
        self.init_oracle_trainng()
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.log = open('experiment-log-mle.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        self.init_metric()

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                self.evaluate()
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        return


    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token, add_data = self.add_data)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = None

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict

    def train_real(self, data_loc=None):
        from utils.text_process import code_to_text, text_to_code
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
        self.init_metric()

        # def file_to_code(wi_dict=wi_dict, data_loc=None, seq_length=self.sequence_length):
        #     tokens = get_tokenlized(data_loc)
        #     out_path = data_loc.strip('.txt') + '_code.txt'
        #     with open(out_path, 'w') as outfile:
        #         outfile.write(text_to_code(tokens, wi_dict, seq_length))
        #     return out_path
        #self.g_x_file = file_to_code(data_loc=self.g_x_file)

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        def my_pre_train_epoch(sess, trainable_model, data_loader, data_loader_grade, data_loader_class):
            # Pre-train the generator using MLE for one epoch
            supervised_g_losses = []
            data_loader.reset_pointer()
            data_loader_grade.reset_pointer()
            data_loader_class.reset_pointer()

            for it in range(data_loader.num_batch):
                batch = data_loader.next_batch()
                batch_grade = data_loader_grade.next_batch()
                batch_class = data_loader_class.next_batch()
                _, g_loss = trainable_model.pretrain_step(sess, batch, batch_grade, batch_class)
                supervised_g_losses.append(g_loss)

            return np.mean(supervised_g_losses)

        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 30
        #self.adversarial_epoch_num = 100
        self.log = open('experiment-log-mle-real.csv', 'w')
        #my_generate_samples(self.sess, self.generator, self.batch_size, self.sequence_length, self.generate_num, self.generator_file, g_x_file=self.g_x_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        if(self.add_data == True):
            self.data_loader_grade = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
            self.data_loader_class = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
            self.data_loader_grade.create_batches(data_file=self.grade_file)
            self.data_loader_class.create_batches(data_file=self.class_file)

        saver=tf.train.Saver()
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            if(self.add_data == True):
                loss = my_pre_train_epoch(self.sess, self.generator, self.gen_data_loader, self.data_loader_grade, self.data_loader_class)
            else:
                loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            print('Now the loss is %.6f'%loss)
            self.add_epoch()
            if epoch % 5 == 0:
                my_generate_samples(self.sess, self.generator, self.batch_size, self.sequence_length, self.generate_num, self.generator_file,
                                    g_x_file=self.g_x_file, g_grade_file=self.g_grade_file, g_class_file=self.g_class_file)
                get_real_test_file()
                #self.evaluate()
                saver.save(self.sess, 'ckpt/lstm.ckpt')

        saver=tf.train.Saver()
        model_file=tf.train.latest_checkpoint('ckpt/')
        saver.restore(self.sess, model_file)
        my_generate_samples(self.sess, self.generator, self.batch_size, self.sequence_length, self.generate_num, self.generator_file,
                            g_x_file=self.g_x_file, g_grade_file=self.g_grade_file, g_class_file=self.g_class_file)
        get_real_test_file()



