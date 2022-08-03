from trainingpoints import pk_linear


def main(nlhs: list):

    for i in nlhs:
        print(f"Generating training points for {i} LH samples")
        cosmos, pkl = pk_linear("lhs_" + str(i), redshift=0.0)


if __name__ == "__main__":
    main([600, 800, 900, 1000])
