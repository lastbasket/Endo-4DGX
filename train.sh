PYTHONPATH='.'  python train.py -s data/endonerfec/pulling_soft_tissues --port 6019 --expname "endonerf/pulling"  --configs arguments/endonerf/pulling_mono.py 
# PYTHONPATH='.'  python train.py -s data/endonerfec/cutting_tissues_twice --port 6017 --expname "endonerf/cutting"  --configs arguments/endonerf/cutting_mono.py 
# PYTHONPATH='.'  python train.py -s data/c3vd/cecum_t2_b --port 6017 --expname "c3vd/cecum_t2_b"  --configs arguments/c3vd/cecum_t2_b.py 
# PYTHONPATH='.'  python train.py -s data/c3vd/sigmoid_t2_a --port 6017 --expname "c3vd/sigmoid_t2_a"  --configs arguments/c3vd/sigmoid_t2_a.py 
# PYTHONPATH='.'  python train.py -s data/stereomis/P1_1 --port 6017 --expname "stereomis/P1_1"  --configs arguments/stereomis/P1_1.py 
# PYTHONPATH='.'  python train.py -s data/stereomis/P1_2 --port 6017 --expname "stereomis/P1_2"  --configs arguments/stereomis/P1_2.py 
