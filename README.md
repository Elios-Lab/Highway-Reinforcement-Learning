Il progetto riguarda l'applicazione del Reiforcement Learning su ambienti di simulazione riguardanti la guida autonoma.
Si basa sul lavoro di Edouard Leurent dal titolo "An Enviroment for Autonomoud Driving Decision-Making" .

@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}

Il progetto ha due obiettivi: cercare di rendere l'environment più fedele alla realtà aumentando le reward e le regole che la macchina deve seguire e
di analizzare, attraverso l'uso di tabelle Excel, la differenza tra tre diverse tipologie di "intelligenze" della macchina:
-) Reinforcement Learning
-) Random actions
-) Heuristic Actions

Per nostro progetto è stato modificato l'environment Highway_env.py: portandolo a tre corsie e modificando la funzione di reward. 
Sono state create 5 nuove classi:
-) RL_policy_env.py : si tratta 3 corsie highway. A differenza del highway_env.py da cui prende spunto, sono state aggiunte delle reward per migliorare il comportamento della macchina facendole rispettare le regole stradali come limite di velocità, distanza di sicurezza, divieto di sorpasso a destra.
-) Random_policy_ev.py: la differenza dal RL_policy_env.py sta nel fatto che la macchina utilizza azioni scelte a random per muoversi.
-) Heuristic_policy_env.py: la differenza dai due Enviroment precedenti è data dall'utilizzo di azioni precedentemente decise che la macchina segue quando si trova in determinate situazioni.
-) Json.py: classe utilizzata per caricare i fil Json e per verificare la loro validità con lo schema di default
-) statistic.py : classe che viene richiamata da tutti e tre gli environment. Metodi creati per per  creare le tabelle su excel

Inoltre, sono stati utilizzate anche le classi:
-)	Abstract.py (dentro alla cartella env/common) : in particolare il metodo “step”, richiamato sia per il training che per il testing
-)	Road.py (dentro alla cartella road) : richiamando il metodo “neighbour_vehicles” in Behaviuor
-)	Behavior.py (dentro alla cartella vehicle): sono stati creati i metodi “overtaking”, "leftovertaking", “safedistance”, "hazardous", "safedistance_TTC", "hazardous_lane_change_TTC" "get_heuristic_action" e "get_random_action" 

Nota: per creare/modificare il file excel bisogna modificare il percorso del PATH su google colab (os.environ['HIGHWAY_ENV_PATH'] = '/content/drive/MyDrive/..')
N.B Per avere buoni risultati il training di almeno almeno 150000 step

Istruzioni:
File .ipynb: da far girare su colab per vedere video e produrre l’excel con le statistiche
	-)RL_env: per la policy RL. Per farlo andare bisogna mettere la cartella highway-env-master sul proprio account google drive. 
		La seconda cella chiede il permesso ad accedere al proprio account google. 
		Nella quarta cella installa la cartella highway-env-master su colab: bisogna inserire il percorso file della cartella, non il percorso file dal pc ma quello della cartella su colab. 
		(es. /content/drive/MyDrive/..)
		Nell’ottava cella ci sono model.learn e model.save che servono per il train. Model=DQN.load serve per caricare lo zip contenente un train già creato 
		(per caricarlo, bisogna modificare il percorso file con il proprio di colab, come per l’install di highwawy_env_master).
		Utilizzare: "env.config["env"] = 12" in fase di training e "env.config["env"] = 0" in fase di Testing
	-)Random_env: per la policy random.
	-)Heuristic_env: per la policy Euristica.


Moduli Utilizzati:

•	Behavior: al suo interno troviamo la classe “IDMVehicle” ovvero la classe che controlla le azioni longitudinali e laterali di ogni veicolo:

	o	Longitudinali: il modello IDM calcola un'accelerazione data la distanza e la velocità del veicolo che lo precede.

	o	Laterali: il metodo MOBIL decide quando cambiare corsia massimizzando l'accelerazione dei veicoli vicini.

Questa classe non viene solo utilizzata dall’ego_vehicle, ma anche da ogni veicolo presente nell’ambiente. 
Infatti, il sopracitato metodo “MOBIL” viene richiamato solo dagli altri veicoli quando hanno intenzione di 
cambiare corsia (viene utilizzato principalmente per evitare che quetsti veicoli collidano tra loro). 
Completata la panoramica generale della classe “IDMVehicle”, ci focalizziamo sui metodi implementati da noi 
per poter controllare le azioni del veicolo:

	-	rightovertaking: questo metodo è stato il primo che abbiamo implementato, in quanto il primo 
		problema che abbiamo voluto risolvere è stato proprio quello dell’evitare che le macchine facessero
		dei sorpassi a destra, ed è quello che ha richiesto più tempo (in quanto non avevamo ancora ben 
		chiare tutte le dinamiche e le caratteristiche del progetto di Leurent). 
		Al suo interno, viene richiamato il metodo "neighbour_vehicles": questo metodo appartiene alla classe “Road”;
		viene utilizzato in diversi contesti per identificare i due veicoli, quello che segue e quello che precede,
		un determinato veicolo in una determinata corsia. Abbiamo utilizzato "neighbour_vehicles" in modo tale da
 		farci restituire il veicolo che nella corsia a sinistra segue il veicolo Ego. Per capire se l’auto identificata
		da “neighbour_vehicles” sia effettivamente una macchina che è stata superata sulla destra dal nostro ego_vehicle
		abbiamo aggiunto un vincolo: l’ego_vehicle deve andare ad una velocità maggiore dell’altro veicolo. 
		Inoltre, per fare in modo che questo metodo sia circoscritto solo ad intorno ad veicolo superato abbiamo
		aggiunto che la differenza di posizione lungo l’asse X dei due veicoli debba essere minore di 25 metri; 

	-	leftovertaking: il metodo in questione è implementato con un algoritmo molto simile a quello usato per 
		il sorpasso a destra. In quanto, l’unica differenza è data del veicolo restituito dal metodo “neighbour_vehicles”:
		in questo caso è quello che precede l’ego_vehicle nella corsia alla sua destra;
	
	-	safedistance e hazardous: per superare il problema della distanza di sicurezza e del “taglio” della
		traiettoria di un altro veicolo sono stati implementate 4 metodi differenti: i primi due “safedistance” e “hazardous”. 
		Anche queste due funzioni utilizzano il metodo “neighbour_vehicles” per identificare il veicolo che precede, nel caso
		del “safedistance”, e quello che segue, nel caso di “hazardous”, il nostro ego_vehicle ad ogni step. 	
		Anche in questo caso è stato applicato il vincolo della velocità per superare il problema che sia effettivamente
		l’ego_vehicle il problema di tale mancanza di sicurezza. Inoltre, in entrambi i casi, si è posto il vincolo 	
		che la distanza di sicurezza minima da rispettare sia di 8 metri. In caso di non rispetto della distanza minima, 
		la funzione restituisce un valore, che normalizzato, viene moltiplicato con le rispettive reward. Ma questi 	
		non sono stati ritenuti come due metodi ottimali in quanto non tengono conto delle velocità dei veicoli in gioco;   
              
	-	safedistance_TTC e hazardous_lane_change_TTC: per superare il problema dei metodi “safedistance” e “hazardous”
		si è deciso di implementare un algoritmo chiamato Time To Collision (TTC), comunemente usato oggi nel campo 
		automotive. Il TTC è il tempo necessario affinché due veicoli entrino in collisione se continuano alla velocità
		attuale e sullo stesso percorso. Anche in questo caso il valore restituito dalle due funzioni viene, dopo 
		una normalizzazione, moltiplicato con le rispettive reward.

•	statistics: il modulo “statistics” è stato pensato ed implementato principalmente per due motivi: 

	o	collegamento tra i moduli delle varie policy create e 

	o	raccoglitore di dati ad ogni step e ad ogni episodio, per poi utilizzarli per ottenere delle statistiche sulla qualità 
		della simulazione e quindi anche del rispettivo addestramento.

	Per fare ciò si è divisa la classe “Statistics”, contenuta nell’omonimo modulo, in 5 funzioni:

	-) liste: la funzione di collegamento tra i moduli delle policy e la classe IDMVehicle. 
	Viene richiamata ad ogni step dal metodo “_reward” in ogni policy utilizzata. 
	Quando richiamata vengono passati quattro Input, ovvero: 
		o	self: istanza della classe data;
		o	Ego_vehicle: il nostro agente;
		o	Action: l’ultima azione dell’ego_vehicle;
		o	Aglo_distance: l’algoritmo scelto per valutare la distanza di sicurezza.

	All’interno di questa funzione troviamo diverse chiamate a metodi contenuti in IDMVehicle e la raccolta
	di diverse informazioni real time riguardanti caratteristiche del veicolo. Tutti questi dati vengono raccolti in liste
 	personalizzate, ad ogni step, per poi essere analizzati al termine di ogni singolo episodio, da specificare che questo
	avviene solo per la fase di simulazione. Tra le task di “liste” troviamo:
		o	rightovertaking: chiamata a funzione della lista descritta all’inizio del sotto paragrafo, 
			oltre al valore della variabile che stabilisce se si è superato un veicolo sulla destra o meno, 
			si ha anche la raccolta dell’identificativo di suddetto veicolo. Questo servirà in fase statistica per tenere
			traccia più facilmente del numero delle macchine superate in tale modo, infatti ad ogni step l’eventuale ID
			del veicolo superato viene confrontato con gli ID già raccolti nel corso dell’episodio, e viene accodato alla
			lista solo nel caso sia la prima volta che esso appare;
		o	leftovertaking: come per il sorpasso a destra, all’interno di “liste” viene anche richiamata la funzione 
			atta a verificare un possibile sorpasso a sinistra. Anche in questo caso viene raccolto l’ID del veicolo superato: 
			al contrario di rightovertaking, l’ID del veicolo superato sulla sinistra ha non solo una funzione statistica ma viene 
			utilizzato per fare in modo che la reward del sorpasso venga attribuita una sola volta a veicolo. 
			E’ stato scelto questo approccio in quanto si è notato che, senza, l’ego_vehicle iniziava a viaggiare 
			continuamente in corrispondenza dell’area, del primo veicolo che superava, presso cui riceve la ricompensa positiva;
		o	lane: ad ogni step viene raccolto l’identificativo della corsia su cui viaggia l’ego_vehicle. 
			Questo avviene sia a fini statistici e sia perché, a seconda della corsia, il metodo ritorna il valore 
			che poi viene moltiplicato per la reward “Right_Lane_Reward”; 
		o	algoritmo per la distanza di sicurezza: a seconda del valore di “distance_algo”, 
			il metodo statistics richiama le funzioni per la verifica del mantenimento della distanza di sicurezza.
			Inoltre, una volta che viene riscontrato il mancato rispetto della norma, viene calcolata il valore delle variabili
			che moltiplicano le reward “Unsafe_Distance_Reward” e “Hazardous_Lane_Change_Reward”. 
			In particolare, queste variabili vengono calcolate in modo che più il nostro veicolo si avvicina all’altro
			veicolo e più il valore della variabile è alto. Infine, il valore delle due distanze di sicurezza, ad ogni step, 
			viene raccolto in due specifiche liste per poi utilizzare i dati a fini statistici.
		o	velocità: ad ogni step viene controllata la velocità del nostro ego_vehicle e viene verificato in quale range
			di valori si trova: se la velocità fosse inferiore a 29 m/s la funzione ritorna un valore che aumenta più si discosta 
			negativamente dal valore di soglia; se la velocità fosse compresa tra 29 m/s e 30 m/s la funzione ritorna un valore pari a 1; 
			infine se il valore della velocità fosse maggiore di 30 m/s la funzione ritorna un valore che aumenta più si discosta dal valore di soglia. 
			In quest’ultimo caso il fattore pesa maggiormente rispetto al caso in cui la velocità sia inferiore a 29 m/s:
			questo per far capire all’agente che è più “sbagliato” superare i 30 m/s che rallentare. 
			Bisogna specificare che solo il range di valori tra 29 m/s e 30 m/s viene moltiplicato per reward positiva della velocità 
			(High_Speed_Reward), gli altri due range vengono moltiplicati per la reward negativa della velocità (Speed_Reward).
		o	accelerazione, decelerazione e steering: come nel caso delle velocità, ad ogni step viene controllato se l’ego_vehicle accelera, 
			decelera o sterza. A differenza del caso della velocità non esistono dei range che differenzino in casi le tre proprietà.  
			Da segnalare che sono per un fattore di normalizzazione, lo steering viene moltiplicato per un fattore 10. 
			Tutte e tre le proprietà vengono moltiplicate in “_reward” per la rispettiva reward negativa.

	-) Statistics: l’obiettivo di questa funzione è quello di esaminare i dati raccolti al termine di ogni episodio, nella fase di testing. 
	Ad ogni metrica che ne necessita, infatti, è associata una lista con i dati raccolti durante ogni step del relativo episodio. 
	Una volta completata l’analisi per ogni singola metrica, i risultati vengono salvati e accodati ad una lista chiamata “list_evaluate”.
	Questa lista viene caricata al termine di ogni episodio dai risultati delle metriche di ciascun episodio e 
	solo una volta completati tutti gli episodi, viene passata ad altri due metodi, che descriveremo in seguito: “evaluateEP” e “printEpisodes”.
	
	-) evaluateEP: creato per salvare e per stampare successivamente su Excel, lo score di ogni singolo episodio. 
	Infatti, riceve in ingresso la “list_evaluate” contenente le statistiche dell’ultimo episodio e le confronta 
        con le soglie scelte e definite nel file json “paramsEpisode.json” per poi calcolare il punteggio di ogni singola 
	metrica e ritornare il valore dello score totale. Tale valore viene anch’esso accodato a “list_evaluate” in statistics, 
	per poi essere letto in fase di creazione del file Excel.

	-) printEpisodes: il metodo principale per la creazione del file Excel. Infatti, inizia caricando il file Excel 
	precedentemente creato e chiamato, nel nostro caso, “evaluationPolicies.xlsx”. Il percorso del file viene definito nel file Jupyter
	grazie alla funzione “os”. All’interno del file sono presenti tre fogli: 
		o	RL Policy: dove vengono stampate le statistiche della policy riguardante il Reinforcement Learning;
		o	Random Policy: dove vengono stampate le statistiche riguardanti la policy randomica;
		o	Heuristic Policy: dove vengono stampate le statistiche della policy euristica.
	A questo punto, attraverso dei cicli for ogni elemento di “list_evaluate” viene stampato in una cella del foglio Excel.
	Successivamente, viene richiamato il metodo “evaluatePolicy” che ritorna il valore dello score dell’episodio da un’ora e tre liste:
	una contenente le statistiche dell’episodio da un’ora che viene poi stampato sul file Excel; una lista contenente il punteggio 
	per ogni metrica e una lista contenente la percentuale di successo per ogni metrica. Infine, viene salvato il file Excel.

	-) evaluatePolicy: il metodo in questione è stato creato per valutare l’episodio complessivo da un’ora. 
	Ad inizio metodo vengono caricati i due file Json riguardanti le soglie e il peso di ogni metrica. 
	Successivamente, per ogni metrica vengono valutati e comparati i risultati di ogni singolo episodio, 
	per poi confrontarli con le soglie scelte e calcolare il punteggio di ogni metrica. Il metodo termina col calcolo dello score totale.

Nel progetto sono stati utilizzati quattro differenti json:
-) params.json : utilizzato per impostare i parametri degli environment Random e Heuristic. Di seguito il json di default.

{
    "lanes_count": 8,
    "vehicles_count": 10,
    "initial_lane_id": 2,
    "duration": 120, 
    "ego_spacing": 0.5, 
    "vehicles_density": 0.7,
    "initial_speed_ego": 25
}

-) paramsEvaluationFunction.json: Utilizzato per impostare quali statistiche ci si aspetta dalla simulazione. Di seguito il json di default.

{
    "max_crush_per_h": 4,
    "max_km_per_h": 110,
    "max_right_overtake_per_h": 10,
    "max_left_overtake_per_h": 150,
    "max_AVG_TTC_per_h": 10,
    "max_STD_TTC_per_h": 10,
    "max_AVG_DSL_per_h": 5,
    "max_STD_DSL_per_h": 5,
    "max_AVG_LOAF_per_h": 2,
    "max_number_lane_change_per_h": 100,
    "max_hazardous_per_h": 10,
    "max_AVG_ACC_per_h": 0.5,
    "max_STD_ACC_per_h": 0.5,
    "max_AVG_DEC_per_h": 0.1,
    "max_STD_DEC_per_h": 0.1,
    "max_AVG_STER_per_h": 0.1,
    "max_STD_STER_per_h": 0.1
}

-) paramsRL.json: utilizzato per impostare i parametri dell'environment RL_policy_env.py e il valore delle reward. Di seguito il json di default.

{
    "lanes_count": 3,
    "vehicles_count": 10,
    "initial_lane_id": 2,
    "duration": 120, 
    "ego_spacing": 0.5, 
    "vehicles_density": 0.7,
    "collision_reward": -1,  
    "limit_speed": 30,
    "initial_speed_ego": 30,
    "High_Speed_Reward": -0.4,
    "Right_Overtaking_Reward": -0.8,
    "Unsafe_Distance_Reward": -0.4,
    "Right_Lane_Reward" : -0.4,
    "Left_Overtaking_Reward" : 0.9,
    "Long_Accel_Reward" : -0.1,
    "Long_Decel_Reward" : -0.1,
    "Hazardous_Lane_Change_Reward" : -0.5,
    "Steering_Reward" : -0.1,
    "SafeDistance" : 1 
}

-) weightparamsEvaluationFunction.json: utilizzato per impostare i pesi di ciascuna metrica del paramsEvalluationFunction.json. Di seguito il json di default.

{
    "weight_crush": 15,
    "weight_km": 20,
    "weight_right_overtake": 6,
    "weight_left_overtake": 11,
    "weight_AVG_TTC": 5,
    "weight_STD_TTC": 2,
    "weight_AVG_DSL": 5,
    "weight_STD_DSL": 3,
    "weight_AVG_LOAF": 8,
    "weight_number_lane_change": 5,
    "weight_hazardous": 8,
    "weight_AVG_ACC": 2,
    "weight_STD_ACC": 2,
    "weight_AVG_DEC": 2,
    "weight_STD_DEC": 2,
    "weight_AVG_STER": 2,
    "weight_STD_STER": 2
}




