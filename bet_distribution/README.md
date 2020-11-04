# Vyhodnocení sázek na základě pravděpodobností

## vstupy
(v jedné iteraci - pomocí incrementálních dat)

a) pravděpodobnosti - _výhra_, _remíza_, _prohra_ (určené námi v jiné části modelu)

  - pro všechny zápasy různé

b) možnosti sázek - `bankroll`, `min_bet` (`summary`), `max_bet` (`summary`)

  - jednotné pro jeden data increment

## podúlohy
1) určení rozdělení poměru vsazené hodnoty (určené námi) mezi sázkařské příležitosti (`opps`).
2) určení optimální hodnoty v dané iteraci
  - bude záležet na významnosti pravděpodobností oproti předpokládaným dalším iteracím.
