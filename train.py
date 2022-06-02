import libcygtrn
from libcygtrn import CYCLES
import yaml

with open("config.yml","r") as f:
    settings = yaml.safe_load(f)

cyc = CYCLES()

cyc.train(cuda = settings["UseCUDA"],dataroot = settings["root"],dataset_p = settings["set"],image_size=settings["i_size"],batch_size=settings["bs"],epochs=settings["ep"],decay_epochs = settings["ep"] / 2,print_freq=settings["pf"])
