from extract_data import get_data
import dnest4.classic as dn4
import subprocess

subprocess.run(["mkdir", "output"])

for i in range(190):
    for band in ["g"]:
        get_data(i, band, deredshift=True, sanitise=True, plot=False)
        subprocess.run("./main -s {seed}".format(seed=i), shell=True)
        dn4.postprocess(plot=False, rng_seed=0)
        subprocess.run(["mv", "posterior_sample.txt",
                        f"output/posterior_sample_{i}_{band}.txt"])

