"""Create boxplots for argument strength vs. emotion."""
import pandas
import scipy.stats
import seaborn
from matplotlib import pyplot as plt


def _normalize_emotions(emotion: bool) -> str:
    if emotion:
        return "emotional"
    return "non-emotional"


def main():
    df = pandas.read_csv("data/master_files/out/arg_master_test_ukp_fix.csv")
    print(len(df))
    # normalize dataset name
    order_ = ["non-emotional", "emotional"]
    rename = {
        "toledo": "IBM-ArgQ",
        "gleize": "IBM-ArgQ",
        "gretz": "IBM-Rank",
        "swanson_dp": "SwanRank",
        "swanson_gm": "SwanRank",
        "swanson_gc": "SwanRank",
        "swanson_evo": "SwanRank",
        "swanson": "SwanRank",
        "UKPConvArgRank": "UKPConvArgRank",
    }
    df["dataset"] = df["dataset"].apply(rename.__getitem__)

    order = ["IBM-ArgQ", "IBM-Rank", "SwanRank", "UKPConvArgRank"]
    column = "label"

    # normalize emotional
    df[column] = df[column].apply(_normalize_emotions)
    color = ["Blue", "Red"]
    seaborn.boxplot(x=df.dataset, y=df.arg_strength, data=df, hue=df.label, hue_order=order_, order=order,
                    showmeans=True,
                    meanprops={"markeredgecolor": "white", "markersize": "7", "markerfacecolor": "black"},
                    palette=color)

    p = {}
    for dataset, group in df.groupby(by="dataset"):
        vs = []
        for _, sub_group in group.groupby(by=column):
            vs.append(sub_group["arg_strength"].values)
        assert len(vs) == 2
        p[dataset] = scipy.stats.ttest_ind(*vs, equal_var=False).pvalue
        print(p[dataset])

    plt.legend(loc="lower right")
    plt.title('Human annotation & argument strength')
    plt.ylabel('Argument strength')
    plt.xlabel('Datasets')

    plt.show()


if __name__ == "__main__":
    main()
