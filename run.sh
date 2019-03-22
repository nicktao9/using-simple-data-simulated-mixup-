python generate.py generate_train_data --display True
python generate.py generate_test_data --display True
python main.py main --mixup "$1"    #
python plot.py generate_fig --mixup "$1"  # 