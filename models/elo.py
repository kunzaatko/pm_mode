#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: Toto je naprosto optimální příklad na třídu. Ustálil by se i API <09-11-20, kunzaatko> #
# FIXME: prosím doplnit komentáře. Dost toho je špatně k pochopení, pokud nejsi autorem kódu. <09-11-20, kunzaatko> #

# DataFrame z celé historie zápasů
df = pd.read_csv("/home/sramon/fotbal_data.csv")

# úprava DataFrame
def c_elo(df):
    # TODO: Tohle nebude fungovat, protože nebudeme mít k dispozici všechna data <08-11-20, kunzaatko> #
    elo = pd.DataFrame(
        np.zeros((len(df["Date"].unique()) + 1, len(df["AID"].unique())))
    )

    # definuj unikátní týmy
    for i, team in enumerate(df["HID"].unique(), start=0):
        elo.rename(columns={i: f"{team}"}, inplace=True)

    # FIXME: nerozumím, proč se tohle dělá... Proč se kromě unikátního týmu hledá i unikátní datum hry? Však tím se vyřadí týmy, které začnou hrát ve stejné datum a pokud ne tak je to zbytečné... Vysvětlit v komentáři, prosím <09-11-20, kunzaatko> #
    for i, date in enumerate(df["Date"].unique()):
        elo.rename(index={i: f"{date}"}, inplace=True)

    # přidání základního ela
    for team in elo:
        elo.at[0, f"{team}"] = 1500
    elo.rename(index={0: "0"}, inplace=True)
    return elo


# aktualizace ela dvou týmu po zápase # FIXME: takové komenáře do dokumentace funkce [https://en.wikibooks.org/wiki/Python_Programming/Source_Documentation_and_Comments] &1 <09-11-20, kunzaatko> #
# FIXME: prosím, komentáře k inputům funkce. Není jasné zda to jsou booly a co přesně vyjadřují v rámci příkladu. Co je `K`? atd. <09-11-20, kunzaatko> #
def update_elo(index, K, home_team, away_team, elo, home_win, away_win, draw, today):
    mean_elo = 1500  # FIXME: nepoužito <08-11-20, kunzaatko> #
    k_factor = 20
    draw_factor = 0

    if draw == 1:
        draw_factor = 0.5

    last_match_ofhome = elo[elo[f"{home_team}"] != 0].index[-1]
    last_match_ofaway = elo[elo[f"{away_team}"] != 0].index[-1]

    elo_home_team = elo.at[f"{last_match_ofhome}", f"{home_team}"]
    elo_away_team = elo.at[f"{last_match_ofaway}", f"{away_team}"]

    koeficient = (
        int(elo_away_team) - int(elo_home_team)
    ) / 400  # FIXME: nepoužito <08-11-20, kunzaatko> #

    liga = df.at[index, "LID"]

    expected_win_HOME = (
        1 / (1 + np.int32(10) ** ((int(elo_away_team) - int(elo_home_team)) / 400))
        + K.at[f"{liga}", "values"]
    )
    expected_win_AWAY = (
        1 / (1 + np.int32(10) ** ((int(elo_home_team) - int(elo_away_team)) / 400))
        - K.at[f"{liga}", "values"]
    )

    elo_home = elo_home_team + k_factor * (home_win + draw_factor - expected_win_HOME)
    elo_away = elo_away_team + k_factor * (away_win + draw_factor - expected_win_AWAY)

    return elo_home, elo_away


# FIXME: &1 <09-11-20, kunzaatko> #
# výroba tabulky el row = týden, column = tým, postfixově k datu # TODO: &1 <09-11-20, kunzaatko> #
def create_elo(df, K):
    elo = c_elo(df)
    yesterday = "0"
    for index in df.index:
        today = df.at[index, "Date"]
        if yesterday != today:
            for team in elo:
                elo.at[f"{today}", f"{team}"] = elo.at[f"{yesterday}", f"{team}"]
        yesterday = today
        home_team = df.at[index, "HID"]
        away_team = df.at[index, "AID"]
        home_win = df.at[index, "H"]
        draw = df.at[index, "D"]
        away_win = df.at[index, "A"]
        (
            elo.at[f"{today}", f"{home_team}"],
            elo.at[f"{today}", f"{away_team}"],
        ) = update_elo(
            index, K, home_team, away_team, elo, home_win, away_win, draw, today
        )

    return elo


# FIXME: Jaký je rozdíl od create_elo? Docs. <09-11-20, kunzaatko> #
def create_elo2(elo):
    elo2 = elo

    for i, date in enumerate(df["Date"].unique()):
        elo2.rename(index={f"{date}": i}, inplace=True)
    elo2.rename(index={"0": 0}, inplace=True)

    for i, date in enumerate(df["Date"].unique(), start=0):
        elo.rename(index={i: f"{date}"}, inplace=True)
    return elo2


def expected_wins(K, home_team, away_team, df, index):
    elo_home_team = df.at[index, "elo_Home"]
    elo_away_team = df.at[index, "elo_Away"]
    liga = df.at[index, "LID"]
    koeficient = (
        int(elo_away_team) - int(elo_home_team)
    ) / 400  # FIXME: nepoužito <08-11-20, kunzaatko> #
    expected_win_HOME = (
        1 / (1 + np.int32(10) ** ((int(elo_away_team) - int(elo_home_team)) / 400))
        + K.at[f"{liga}", "values"]
    )
    expected_win_AWAY = (
        1 / (1 + np.int32(10) ** ((int(elo_home_team) - int(elo_away_team)) / 400))
        - K.at[f"{liga}", "values"]
    )
    return expected_win_HOME, expected_win_AWAY


def add_to_df(df, elo2):
    for index in df.index:
        today = df.at[index, "Date"]
        HID = df.at[index, "HID"]
        AID = df.at[index, "AID"]
        df.at[index, "elo_Home"] = elo2.at[f"{today}", f"{HID}"]
        df.at[index, "elo_Away"] = elo2.at[f"{today}", f"{AID}"]


def expected_r(K, df):
    expected = np.zeros((len(df), 2))
    for index in df.index:
        home_team = df.at[index, "elo_Home"]
        away_team = df.at[index, "elo_Away"]
        date = df.at[index, "Date"]  # FIXME: nepoužito <08-11-20, kunzaatko> #
        expected_win_HOME, expected_win_AWAY = expected_wins(
            K, home_team, away_team, df, index
        )

        add = np.array([expected_win_HOME, expected_win_AWAY])

        expected[index, :] = add
    return expected


def compare(expected, j):

    if (expected[0] - expected[1]) > j:
        return np.array([1, 0, 0])

    elif (expected[1] - expected[0]) > j:
        return np.array([0, 0, 1])
    else:
        return np.array([0, 1, 0])


def win_ratio_home(df, LD):
    liga = df[df["LID"] == f"{LD}"]
    wins_home = len(liga[liga["H"] == 1])
    matches = len(liga)
    win_ratioH = wins_home / matches
    return win_ratioH


def draw_ratio(df, LD):
    liga = df[df["LID"] == f"{LD}"]
    draws = len(liga[liga["D"] == 1])
    matches = len(liga)
    draws_ratios = draws / matches
    return draws_ratios


# TODO: Do jiného souboru? <09-11-20, kunzaatko> #
####################################
#  Konec funkcí týkajících se ELO  #
####################################

def win_ratio_away(df, LD):
    liga = df[df["LID"] == f"{LD}"]
    wins_away = len(liga[liga["A"] == 1])
    matches = len(liga)
    win_ratioA = wins_away / matches
    return win_ratioA


def ratio_per_LID(df):
    ratio = pd.DataFrame(
        {
            "win_ratio_home": [0.0 for x in df["LID"].unique()],
            "draw_ratio": [0.0 for x in df["LID"].unique()],
            "win_ratio_away": [0.0 for x in df["LID"].unique()],
        },
        index=[x for x in df["LID"].unique()],
    )
    for liga in df["LID"].unique():
        ratio.at[f"{liga}", "win_ratio_home"] = win_ratio_home(df, liga)
        #  print(win_ratio_home(df, liga))
        ratio.at[f"{liga}", "draw_ratio"] = draw_ratio(df, liga)
        ratio.at[f"{liga}", "win_ratio_away"] = win_ratio_away(df, liga)

    return ratio


ratio = ratio_per_LID(df)
K = pd.DataFrame(
    {"values": [0.0 for x in df["LID"].unique()]}, index=[x for x in df["LID"].unique()]
)
for m in range(1, 20):
    elo = create_elo(df, K)
    elo2 = create_elo2(elo)
    add_to_df(df, elo2)
    expected = expected_r(K, df)
    e = [x[0] for x in expected]
    df["expected_winsH"] = pd.DataFrame(e)
    e = [x[1] for x in expected]
    df["expected_winsA"] = pd.DataFrame(e)
    for liga in df["LID"].unique():
        e = df[df["LID"] == liga]
        s = (
            ratio.at[f"{liga}", "win_ratio_home"]
            + 0.5 * ratio.at[f"{liga}", "draw_ratio"]
        )
        K.at[f"{liga}", "values"] = s - e["expected_winsH"].sum() / len(e)
        print(K.at[f"{liga}", "values"])
        print(e["expected_winsH"].sum() / len(e))
        print("_____")
    print("______________________")


ratio = ratio_per_LID(df)
ratio


# for j in range(1, 20):
j = 0
predict = np.zeros((len(df), 3))


expected = expected_r(K, df)
for i,row in enumerate(expected, start=0):
    predict[i, :] = compare(row, j)

correct = 0
results = df[["H", "D", "A"]].to_numpy()
for i,(rowr, rowp) in enumerate(zip(results, predict)):
    if (rowr == rowp).all():
        correct += 1

print(correct / len(results))
