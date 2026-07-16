.PHONY: setup train evaluate record verify clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	git clone https://github.com/google-deepmind/mujoco_menagerie.git || true

verify:
	python scripts/verify_model.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py --render

record:
	python scripts/record_video.py

tensorboard:
	tensorboard --logdir logs/tensorboard/

clean:
	rm -rf models/go2_ppo_*.zip logs/tensorboard/*
