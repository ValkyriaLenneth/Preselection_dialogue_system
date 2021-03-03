from MALCopyDataset import *
from newModel import *
from torch import optim
from trainers.DefaultTrainer import *
import torch.backends.cudnn as cudnn

if __name__ == '__main__':

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    # Data_path, vocab, dataset
    data_path = 'dataset/holl.256/'
    output_path = 'log_mal/holl.256/'

    # get vocab
    tokenizer, vocab2id, id2vocab = bert_tokenizer()

    # src_vocab2id, src_id2vocab, src_id2freq = load_vocab(data_path + 'holl_input_output.256.vocab', t=10) # t=10 for test
    # tgt_vocab2id, tgt_id2vocab, tgt_id2freq = src_vocab2id, src_id2vocab, src_id2freq

    #train_dataset = MALDataset([data_path + 'holl-train.256.json'], vocab2id, tokenizer)
    train_dataset = MALDataset([data_path + 'holl-dev.256.json'], vocab2id, tokenizer) # for test
    dev_dataset = MALDataset([data_path + 'holl-dev.256.json'], vocab2id, tokenizer)
    test_dataset = MALDataset([data_path + 'holl-test.256.json'], vocab2id, tokenizer)

    # Model and Optimizer
    encoder=BertEncoder()
    selector =Selector(768, encoder)
    generator = Generator(encoder, 768, len(vocab2id))
    model=MAL(encoder, selector, generator, id2vocab, vocab2id, id2vocab, vocab2id, max_dec_len=50, beam_width=1)
    init_params(model)

    # Bert: 2e-5, Model: 1e-3
    bert_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    model_params = []
    for name, param in model.named_parameters():
        if name.find('encoder') != -1:
            continue
        else:
            if param.requires_grad == True:
                model_params.append(param)
    model_optimizer = optim.Adam(model_params, lr=0.01)


    batch_size = 8

    trainer = DefaultTrainer(model)

    for i in range(30):
        trainer.train_epoch('mle_train', train_dataset, train_collate_fn, batch_size, i, model_optimizer)
        rouges = trainer.test(dev_dataset, test_collate_fn, batch_size, i, output_path=output_path)
        rouges = trainer.test(test_dataset, test_collate_fn, batch_size, 100+i, output_path=output_path)
        trainer.serialize(i, output_path=output_path)