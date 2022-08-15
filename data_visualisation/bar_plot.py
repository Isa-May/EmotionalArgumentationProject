import matplotlib.ticker as mtick
import pandas as pd


def main():
    df = pd.read_csv(
        "data/master_files/master_predictions/EmoBERTbin_pred/EmoBERTpredSeed1EntireSet.csv"
    )

    conf_min_60 = df[(df['confidence'] > 0.66) & (df['stance'] != 2)]
    print(conf_min_60)

    df = conf_min_60
    pro_stance_emoBERT = df[df['stance'] == 1]
    contra_stance_emoBERT = df[df['stance'] == -1]
    pro_stance_emo = df[(df['stance'] == 1) & (df['label'] == 1)]
    len_pro_stance_emo = (len(pro_stance_emo) * (1 / len(pro_stance_emoBERT))) * 100
    pro_stance_non_emo = df[(df['stance'] == 1) & (df['label'] == 0)]
    len_pro_stance_non_emo = (len(pro_stance_non_emo) * (1 / len(pro_stance_emoBERT))) * 100
    contra_stance_emo = df[(df['stance'] == -1) & (df['label'] == 1)]
    len_contra_stance_emo = (len(contra_stance_emo) * (1 / len(contra_stance_emoBERT))) * 100

    contra_stance_non_emo = df[(df['stance'] == -1) & (df['label'] == 0)]
    len_contra_stance_non_emo = (len(contra_stance_non_emo) * (1 / len(contra_stance_emoBERT))) * 100

    emo_pro_contra_emoBERT = [len_pro_stance_emo, len_contra_stance_emo]
    emo_pro_contra_emoBERT = [round(num, 2) for num in emo_pro_contra_emoBERT]

    non_emo_pro_contra_emoBERT = [len_pro_stance_non_emo, len_contra_stance_non_emo]
    non_emo_pro_contra_emoBERT = [round(num, 2) for num in non_emo_pro_contra_emoBERT]

    #############
    ###argbert###
    #############

    df = pd.read_csv(
        "data/master_files/master_predictions/argbert_pred/argbert_predSeed3EntireSet.csv"
    )

    conf_min_60 = df[(df['confidence'] > 0.66) & (df['stance'] != 2)]
    print(conf_min_60)

    df = conf_min_60
    pro_stance_argBERT = df[df['stance'] == 1]
    contra_stance_argBERT = df[df['stance'] == -1]
    pro_stance_emo = df[(df['stance'] == 1) & (df['label'] == 1)]
    len_pro_stance_emo = (len(pro_stance_emo) * (1 / len(pro_stance_argBERT))) * 100
    pro_stance_non_emo = df[(df['stance'] == 1) & (df['label'] == 0)]
    len_pro_stance_non_emo = (len(pro_stance_non_emo) * (1 / len(pro_stance_argBERT))) * 100
    contra_stance_emo = df[(df['stance'] == -1) & (df['label'] == 1)]
    len_contra_stance_emo = (len(contra_stance_emo) * (1 / len(contra_stance_argBERT))) * 100

    contra_stance_non_emo = df[(df['stance'] == -1) & (df['label'] == 0)]
    len_contra_stance_non_emo = (len(contra_stance_non_emo) * (1 / len(contra_stance_argBERT))) * 100

    emo_pro_contra = [len_pro_stance_emo, len_contra_stance_emo]
    emo_pro_contra = [round(num, 2) for num in emo_pro_contra]

    non_emo_pro_contra = [len_pro_stance_non_emo, len_contra_stance_non_emo]
    non_emo_pro_contra = [round(num, 2) for num in non_emo_pro_contra]

    ###argbertemoinit
    df = pd.read_csv(
        "data/master_files/master_predictions/argBERTEmoInit_pred/argBERTEmoInitPredSeed5EntireSet.csv"
    )

    conf_min_60 = df[(df['confidence'] > 0.66) & (df['stance'] != 2)]
    print(conf_min_60)

    df = conf_min_60
    pro_stance_argBERTEmoInit = df[df['stance'] == 1]
    contra_stance_argBERTEmoInit = df[df['stance'] == -1]
    pro_stance_emo = df[(df['stance'] == 1) & (df['label'] == 1)]
    len_pro_stance_emo = (len(pro_stance_emo) * (1 / len(pro_stance_argBERTEmoInit))) * 100
    pro_stance_non_emo = df[(df['stance'] == 1) & (df['label'] == 0)]
    len_pro_stance_non_emo = (len(pro_stance_non_emo) * (1 / len(pro_stance_argBERTEmoInit))) * 100
    contra_stance_emo = df[(df['stance'] == -1) & (df['label'] == 1)]
    len_contra_stance_emo = (len(contra_stance_emo) * (1 / len(contra_stance_argBERTEmoInit))) * 100

    contra_stance_non_emo = df[(df['stance'] == -1) & (df['label'] == 0)]
    len_contra_stance_non_emo = (len(contra_stance_non_emo) * (1 / len(contra_stance_argBERTEmoInit))) * 100

    emo_pro_contra_argBERTEmoInit = [len_pro_stance_emo, len_contra_stance_emo]
    emo_pro_contra_argBERTEmoInit = [round(num, 2) for num in emo_pro_contra_argBERTEmoInit]

    non_emo_pro_contra_argBERTEmoInit = [len_pro_stance_non_emo, len_contra_stance_non_emo]
    non_emo_pro_contra_argBERTEmoInit = [round(num, 2) for num in non_emo_pro_contra_argBERTEmoInit]

    # argbertemoinit*2
    df = pd.read_csv(
        "data/master_files/master_predictions/ensemble_pred/ensemble_ArgBeEmoInit1_arg_entire.csv"
    )

    conf_min_60 = df[(df['confidence'] > 0.66) & (df['stance'] != 2)]
    print(conf_min_60)

    df = conf_min_60
    pro_stance_argBERTEmoInitEns = df[df['stance'] == 1]
    contra_stance_argBERTEmoInitEns = df[df['stance'] == -1]
    pro_stance_emo = df[(df['stance'] == 1) & (df['label'] == 1)]
    len_pro_stance_emo = (len(pro_stance_emo) * (1 / len(pro_stance_argBERTEmoInitEns))) * 100
    pro_stance_non_emo = df[(df['stance'] == 1) & (df['label'] == 0)]
    len_pro_stance_non_emo = (len(pro_stance_non_emo) * (1 / len(pro_stance_argBERTEmoInitEns))) * 100
    contra_stance_emo = df[(df['stance'] == -1) & (df['label'] == 1)]
    len_contra_stance_emo = (len(contra_stance_emo) * (1 / len(contra_stance_argBERTEmoInitEns))) * 100

    contra_stance_non_emo = df[(df['stance'] == -1) & (df['label'] == 0)]
    len_contra_stance_non_emo = (len(contra_stance_non_emo) * (1 / len(contra_stance_argBERTEmoInitEns))) * 100

    import matplotlib.pyplot as plt

    emo_pro_contra_argBERTEmoInitEns = [len_pro_stance_emo, len_contra_stance_emo]
    emo_pro_contra_argBERTEmoInitEns = [round(num, 2) for num in emo_pro_contra_argBERTEmoInitEns]

    non_emo_pro_contra_argBERTEmoInitEns = [len_pro_stance_non_emo, len_contra_stance_non_emo]
    non_emo_pro_contra_argBERTEmoInitEns = [round(num, 2) for num in non_emo_pro_contra_argBERTEmoInitEns]

    fig, ax = plt.subplots()

    model_emoBERT = ['EmoBERT pro\n', 'EmoBERT con\n']
    non_emo_emoBERT = [non_emo_pro_contra_emoBERT[0], non_emo_pro_contra_emoBERT[1]]
    emo_emoBERT = [emo_pro_contra_emoBERT[0], emo_pro_contra_emoBERT[1]]

    ax.barh(model_emoBERT, non_emo_emoBERT, color="blue", edgecolor="black")

    ax.barh(model_emoBERT, emo_emoBERT, left=non_emo_emoBERT, color="red", edgecolor="black")

    model = ['ArgBERT pro\n', 'ArgBERT con\n']

    non_emo = [non_emo_pro_contra[0], non_emo_pro_contra[1]]
    emo = [emo_pro_contra[0], emo_pro_contra[1]]

    ax.barh(model, non_emo, color="blue", edgecolor="black")
    ax.barh(model, emo, left=non_emo, color="red", edgecolor="black")

    model_argBERTEmoInit = ['ArgBERT-EmoInit pro\n', 'ArgBERT-EmoInit con\n']

    non_emo_argBERTEmoInit = [non_emo_pro_contra_argBERTEmoInit[0], non_emo_pro_contra_emoBERT[1]]
    emo_argBERTEmoInit = [emo_pro_contra_argBERTEmoInit[0], emo_pro_contra_emoBERT[1]]

    ax.barh(model_argBERTEmoInit, non_emo_argBERTEmoInit, color="blue", edgecolor="black")
    ax.barh(model_argBERTEmoInit, emo_argBERTEmoInit, left=non_emo_argBERTEmoInit, color="red", edgecolor="black")

    model_argBERTEmoInitEns = ['ArgBERT-EmoInit*2 pro\n', 'ArgBERT-EmoInit*2 con\n']
    non_emo_argBERTEmoInitEns = [non_emo_pro_contra_argBERTEmoInitEns[0], non_emo_pro_contra_argBERTEmoInitEns[1]]
    emo_argBERTEmoInitEns = [emo_pro_contra_argBERTEmoInitEns[0], emo_pro_contra_argBERTEmoInitEns[1]]

    ax.barh(model_argBERTEmoInitEns, non_emo_argBERTEmoInitEns, color="blue", edgecolor="black")
    ax.barh(model_argBERTEmoInitEns, emo_argBERTEmoInitEns, left=non_emo_argBERTEmoInitEns, color="red",
            edgecolor="black")
    plt.legend(["Non-emotional", "Emotional"], loc="upper right", edgecolor="black")

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
