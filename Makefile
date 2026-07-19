.PHONY: setup verify train evaluate record compare plot tensorboard clean

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	git clone https://github.com/google-deepmind/mujoco_menagerie.git || true

verify:
	python3 scripts/verify_model.py

train:
	python3 scripts/train.py

evaluate:
	python3 scripts/evaluate.py --episodes 50

evaluate-render:
	python3 scripts/evaluate.py --render --episodes 5

record:
	python3 scripts/record_video.py

compare:
	python3 scripts/compare_algorithms.py

plot:
	python3 scripts/plot_results.py

tensorboard:
	tensorboard --logdir logs/tensorboard/ --port 6006

clean:
	rm -rf models/checkpoints/ models/best/ logs/tensorboard/* logs/eval/
