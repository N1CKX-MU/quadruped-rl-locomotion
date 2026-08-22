.PHONY: help setup check verify train train-scratch resume evaluate evaluate-grid play \
        gait-analysis gait-all record tensorboard test compare clean clean-all

PY ?= venv/bin/python
# On Windows the venv layout differs; override with: make PY=venv/Scripts/python.exe
CONFIG ?= configs/training_config.yaml
RUN ?= go2_ppo
MODEL ?= models/$(RUN)_final.zip

help:
	@echo "Setup"
	@echo "  make setup            create venv, install deps, clone mujoco_menagerie"
	@echo "  make check            10-second environment sanity check (run before training)"
	@echo "  make test             run the unit test suite"
	@echo ""
	@echo "Training"
	@echo "  make train            train with Stable-Baselines3 PPO"
	@echo "  make train-scratch    train with the from-scratch PPO in ppo_from_scratch/"
	@echo "  make resume CKPT=...  resume from a checkpoint (restores VecNormalize stats)"
	@echo "  make tensorboard      open TensorBoard on logs/tensorboard/"
	@echo ""
	@echo "Evaluation"
	@echo "  make play             drive the trained policy by hand (WASD/QE, 1-5 gaits)"
	@echo "  make evaluate         tracking error over random commands"
	@echo "  make evaluate-grid    tracking error swept along each command axis"
	@echo "  make gait-analysis    gait diagram and duty/phase measurement"
	@echo "  make gait-all         one gait diagram per gait"
	@echo "  make record           record a video"
	@echo ""
	@echo "Variables: PY=$(PY)  CONFIG=$(CONFIG)  RUN=$(RUN)  MODEL=$(MODEL)"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git || true
	@echo ""
	@echo "Note: pip installs the CPU build of PyTorch by default. For a GPU:"
	@echo "  pip install torch --index-url https://download.pytorch.org/whl/cu124"

check:
	$(PY) scripts/check_env.py --config $(CONFIG)

verify: check   # scripts/verify_model.py was superseded by check_env.py

test:
	$(PY) -m pytest -q

train:
	$(PY) scripts/train.py --config $(CONFIG) --run-name $(RUN)

train-scratch:
	$(PY) ppo_from_scratch/train_scratch.py --config $(CONFIG)

resume:
	@test -n "$(CKPT)" || (echo "usage: make resume CKPT=models/checkpoints/xxx.zip"; exit 1)
	$(PY) scripts/train.py --config $(CONFIG) --run-name $(RUN) --resume $(CKPT)

tensorboard:
	tensorboard --logdir logs/tensorboard/ --port 6006

play:
	$(PY) scripts/play.py --model $(MODEL)

evaluate:
	$(PY) scripts/evaluate.py --model $(MODEL) --episodes 20

evaluate-grid:
	$(PY) scripts/evaluate.py --model $(MODEL) --grid

gait-analysis:
	$(PY) scripts/gait_analysis.py --model $(MODEL)

gait-all:
	$(PY) scripts/gait_analysis.py --model $(MODEL) --all-gaits

record:
	$(PY) scripts/record_video.py

compare:
	$(PY) scripts/compare_algorithms.py

clean:
	rm -rf models/checkpoints/ models/best/ logs/tensorboard/* logs/eval/

clean-all: clean
	rm -rf models/ __pycache__ */__pycache__ .pytest_cache
