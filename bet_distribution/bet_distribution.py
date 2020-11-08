import numpy as np
import pandas as pd

def inc_bet_distribution(bankroll, P_dis, summary, opps):
    min_bet = summary.iloc[0].to_dict()['Min_bet']
    max_bet = summary.iloc[0].to_dict()['Max_bet']
    P_dis_arr = P_dis.to_numpy # assuming we stick to the suggested data structure as in the input
    odds = opps.to_numpy[7:9,:] # ['OddsH', 'OddsD', 'OddsA']

    # předpokládaný zisk ze sázky jednoho kreditu
    exp_profit_per_cell = P_dis_arr * odds # they should be the same shape

    bets = np.zeros(P_dis_arr.shape()) # underlining structure to count the ideal bets
    purse = bankroll # kolik máme k dispozici kreditů k sázení
    exp_profit = 0 # pro testování
    while purse > (min_bet+max_bet):
        if exp_profit_per_cell.max() > 1.00: # TODO: zkusit různé hodnoty abychom započítaly nepřesnost našeho modelu <08-11-20, kunzaatko> #
            best_bets = np.argmax(exp_profit)
            # for je tu pro případ více cellů se stejným předpokládaným profitem
            for best_bet in best_bets:
                # máme dost na vsazení maximální sázky
                if purse >= max_bet:
                    bets[best_bet] = max_bet
                    exp_profit += max_bet * bets[best_bet]
                    purse -= max_bet
                elif purse >= min_bet:
                    bets[best_bet] = purse
                    exp_profit += max_bet * bets[best_bet]
                    purse -= purse
                    break
                else: # nemáme na minimální sázku
                    break
            exp_profit[best_bets] = 0 # abychom mohli najít další nejvyšší předpokládané zisky, vynulujeme zisky, na které už jsme vsadili
        else:
            break # pokud už není žádný cell, pro který bychom předpokládali zisk

    return pd.DataFrame(data=bets, columns=['BetH', 'BetD', 'BetA'], index=opps.index)

