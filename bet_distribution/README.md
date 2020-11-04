# Vyhodnocení sázek na základě pravděpodobnosti

## vstupy
(v jedné iteraci - pomocí incrementálních dat)

a) pravděpodobnosti - P(_výhra_), P(_remíza_), P(_prohra_) (určené námi v jiné části modelu  P(`H`), P(`D`), P(`A`))

  - pro všechny zápasy různé

b) možnosti sázek - `bankroll`, `min_bet` (`summary`), `max_bet` (`summary`)

  - jednotné pro jeden data increment

c) kurzy - kurz _výhra_, kurz _remíza_, kurz _prohra_ (`opps` (`OddsH`, `OddsD`, `OddsA`))

## podúlohy
1) určení rozdělení poměru vsazené hodnoty (určené námi) mezi sázkařské příležitosti (`opps`).
2) určení optimální hodnoty v dané iteraci
  - bude záležet na významnosti pravděpodobností oproti předpokládaným dalším iteracím.
