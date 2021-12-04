import os
import argparse
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator

import jiant.utils.display as display
import jiant.utils.python.io as py_io


def tokenization(task_name, model_name, control=False, phase=["val"]):
    output_dir = f"./cache/{model_name}/{task_name}"
    if control:
        output_dir = f"./cache/control/{model_name}/{task_name}"
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"probing_tasks/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=model_name,
        output_dir=output_dir,
        phases=phase,
        do_control=control
    ))


def train_configuration(task_name, model_name, classifier_type, do_control=False):
    if do_control:
        task_cache_base_path = f"./cache/control/{model_name}/"
    else:
        task_cache_base_path = f"./cache/{model_name}/"
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="probing_tasks/configs/",
        task_cache_base_path=task_cache_base_path,
        train_task_name_list=[task_name],
        val_task_name_list=[task_name],
        train_batch_size=4,
        eval_batch_size=16,
        epochs=3,
        num_gpus=1,
        classifier_type=classifier_type
    ).create_config()

    os.makedirs("./run_configs/", exist_ok=True)
    py_io.write_json(
        jiant_run_config,
        f"./run_configs/{task_name}_run_config.json")
    display.show_json(jiant_run_config)


def train(
        task_name, model_name, model_path, do_train,
        model_load_mode, freeze_encoder, model_dir_name):
    output_dir = f"./runs/{task_name}/{model_dir_name}/main"
    os.makedirs(output_dir, exist_ok=True)
    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=f"./run_configs/{task_name}_run_config.json",
        output_dir=output_dir,
        hf_pretrained_model_name_or_path=model_name,
        model_path=model_path,
        model_load_mode=model_load_mode,
        model_config_path=f"./models/{model_name}/model/config.json",
        learning_rate=1e-4,
        eval_every_steps=500,
        do_train=do_train,
        do_val=True,
        do_save_best=True,
        write_val_preds=True,
        freeze_encoder=freeze_encoder,
        force_overwrite=True,
        no_cuda=False
    )
    main_runscript.run_loop(run_args)


MODEL_NAMES = {
    "bert1": "bert-base-uncased",
    "bert2": "bert-large-uncased",
    "roberta1": "roberta-base",
    "roberta2": "roberta-large",
    "deberta": "microsoft/deberta-base",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize_train", action="store_true",
                        help="enable tokenization and caching of train data")
    parser.add_argument("--tokenize_val", action="store_true",
                        help="enable tokenization and caching of val data")
    parser.add_argument("--tokenize", action="store_true",
                        help="enable tokenization and caching of both train and val data")
    parser.add_argument("--tokenize_control", action="store_true",
                        help="enable tokenization and caching of control val data")
    parser.add_argument("--main_loop", action="store_true",
                        help="enable train-eval runner"),
    parser.add_argument("--task_name", type=str, default="semgraph2",
                        help="probing task name")
    parser.add_argument("--exp_list", action='append',
                        help="probing experiments name")
    parser.add_argument("--model_name", type=str, default="bert1",
                        help="pre-trained transformer model name")

    args = parser.parse_args()
    task_name = args.task_name
    exp_name = args.model_name
    model_name = MODEL_NAMES[args.model_name] if args.model_name in MODEL_NAMES else ""

    if args.tokenize:
        tokenization(task_name, model_name, phase="train")
        tokenization(task_name, model_name, phase="val")
    elif args.tokenize_train:
        tokenization(task_name, model_name, phase="train")
    elif args.tokenize_val:
        tokenization(task_name, model_name, phase="val")
    elif args.tokenize_control:
        tokenization(task_name, model_name, control=True, phase="train")
        tokenization(task_name, model_name, control=True, phase="val")

    if args.main_loop:
        meta_configs = py_io.read_json(
            f"./run_meta_configs/{task_name}_run_meta_configs.json")
        for exp_nmae in args.exp_list:
            if not exp_name in meta_configs:
                raise KeyError(
                    "Experiment name not found in the meta running configuration!")
        for exp_name in args.exp_list:
            meta_config = meta_configs[exp_name]

            model_path = meta_config["model_pth"]
            model_name = meta_config["model_name"]
            model_val_name = meta_config["model_val_name"]

            do_train = meta_config["do_train"]
            do_control = meta_config["do_control"]
            freeze_encoder = meta_config["freeze_encoder"]
            classifier_type = meta_config["classifier_type"]

            if do_control or not do_train:
                load_mode = "all"
            else:
                load_mode = "from_transformers"

            train_configuration(
                task_name,
                model_name,
                classifier_type=classifier_type,
                do_control=do_control
            )

            train(
                task_name=task_name,
                model_name=model_name,
                model_path=model_path,
                do_train=do_train,
                model_load_mode=load_mode,
                freeze_encoder=freeze_encoder,
                model_dir_name=model_val_name
            )

# python main_probing.py --tokenize  --task_name monotonicity --model_name deberta
# python main_probing.py --tokenize_control --task_name monotonicity --model_name deberta
# python main_probing.py --main_loop --task_name monotonicity --exp_list bert2-mlp --exp_list roberta2 --exp_list roberta2-mlp --exp_list bert2
