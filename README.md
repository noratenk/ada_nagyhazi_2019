# ada_nagyhazi_2019
Solution for Applied Data Analysis Homweork Kaggle Challenge.  
*The following comments are in hungarian, ask me for english version, if necessary.*

# A megoldásom 3 fő része:  
## 1. Előfeldolgozás
*1_preprocess_tuesday2.ipynb*  

Az adatok egy webshop vásárlóinak kattintásait tartalmazzák. Egy-egy kattintássorozatról több sor is van az adathalmazban, 
mivel a kattintások közben többször elmentették a vásárlók kattintásait. Egy kattintássorozat egy session-nek felel meg.  

Az előfeldolgozás 2 része:
* Egy-egy sessionból az utolsó megtartása. Ezen eseményekhez számos új változót defniáltam:
  * click_num: duration_of_session/click_num
  * a kosárban lévő termékek össz ára és a meglátogatott termékek össz árának hányadosa
  * a kosárban lévő legdrágább/legolcsóbb termék árának és a meglátogatott termékek közül a legdrágább/legolcsóbb árának hányadosa
  * a kosárban lévő termékek átlagára
  * új customer value: 'customer value' és 'level_of_purchaing_process' szorzata
  * numerikussá alakított dátum változók (hét napja, óra, perc)
  * adott session-hoz hány "sor" tartozik az eredeti adathalmazban

* Az órai aggregálós módszert követve egy-egy session-hoz a változók minimuma, maximuma és átlaga

Az előfeldolgozás során a hiányzó értékeket nem kezeltem külön, (-1)-el töltöttem fel őket.  
Az előfeldolgozott adathalmaznak végül 82 oszlopa lett.

## 2. Változók szűrése (feature selection)
*2_feature_selection_tuesday2.ipynb*  

A 82 oszlopból kiszűrtem a konstans változókat,  a duplikáltakat, és a nagyon korreláltakat is, majd az eredményt elmentettem. Ez a 
*cust_df_selected_tuesday2.csv*, amely már csak 37 oszlopot tartalmaz.


## 3. Modellezés
*3_model_finetuning_tue_xgb.ipynb*  

A modellem egy xgboost osztályozó modell, amelyetnek a paramétereit keresztvalidáció és GridSearchCV segítségével optimalizáltam. 
Az optimalizálást elvégeztem a szűrt és a teljes adathalmazon is.  

Végül a legnagyobb score értéket ezzel a módszerrel a teljes adathalmazon való modell optimalizálás eredményeként kaptam meg 
(*submission_tue_xgb3.csv*).   

| submission name         | public score | private score |
|-------------------------|--------------|---------------|
| submission_tue_xgb.csv  | 0.89220      | 0.89528       |
| submission_tue_xgb2.csv | 0.89310      | 0.89651       |
| submission_tue_xgb3.csv | 0.89526      | 0.89776       |


*Megjegyzés:*  
A legjobb megoldásomat a submission_tue_xgb3.csv és egy hasonlóan optimalizált RandomForestClassifier modell segítségével készült 
predikció számtani átlagaként számoltam ki. Ez az ensemble eredmény érte el összességében a legjobb, 0.89831 score értéket a private
leaderboradon.


