<h2> Presentation </h2>
Voici une petite api accompagnent le projet de reconnaissance par EEG de l'endormissement au volant.
Cet api vous permet de predire la mise en danger du conducteur sur une base de 17 électrodes a partir d'une sequence Raw_data d'EEG


<h2> Installation </h2>
 1.pré-requis :
 
  - git 
  - docker/docker-compose 
  
```
git clone https://github.com/AurelPuj/EEG_Drowsiness_Driver.git
```


2.lancement du docker-compose :
```bash
cd EEG_Drowsiness_Driver/api/ 
docker-compose up 
```
3.connexion a la base de donnée :

   - ouvrir un nouveau terminal 
   - créer un nouveau user :
   ```bash 
   docker exec -it mongodb bash
   mongo -u mongodbuser -p
   use flaskdb
   db.createUser({user: 'flaskuser', pwd: '1234', roles: [{role: 'readWrite', db: 'flaskdb'}]})
   ```
   - se log sur la base de donnée :
   ``` bash 
   exit
   mongo -u flaskuser -p your password --authenticationDatabase flaskdb
   exit
   exit
   ```
   
    
<h2> Utilisation </h2>

1. Test de l'installation 
  ```bash
  curl -i http://0.0.0.0:5000/version
  ``` 

<h2> Technologie de l'API </h2>

<h4> Principe et shématisation </h4> 
L'utilisateur envoie un fichier contenant les Raw data D'EEG et specifie le modéle à utiliser ( ex: LinearRegression , CNN , Treeboosting ...)
L'API reçoi le fichier et l'instruction du modèle puis en fonction du modèle choisi, l'api vas chercher les poids du modèle dans la base de donée.
Ensuite l'API renvoi les donnée predite dans un fichier json a l'utilisateur.

<img src ="./logo/API_shématic.png">


<h3>Flask API </h3> 

Flask est un framework Web en croissance rapide, conçu pour un processus de conception d'API plus efficace. Eh bien, ce n’est que l’un des usages possibles de Flask.
Flash est léger est très bien documenter. Il exite aussi d'autre framework comme Fast API qui posssede les même avantages que Flask mais nos connaissent préalable sur Flask API nous ont permis de gagner en efficacité sur la mise en place. 
<h3>Mongo DB </h3>

MongoDB est un système de base de donnée orienté objet, dynamique, stable, scalable et sans SQL.
MongoDB execelle dans stockage de document et pour tout objet non lier entre eux hors les poids des modèles sont stockeé dans des fichiers et son totalement independant des autres poids.
Deplus MongoDB effectue très les mise a jour de ses objets ce qui vas arriver frequement dans notre cas. 
Il existe d'autre technologie comme cassandraDB ou postgre mais toute deux moins perfomantes dans la gestion d'objet que mongoDB 

<h3>Docker  </h3>

La technologie nous permet de contenairiser notre API , et de réaliser le liens entre MongoDB et Flask API.
Nous avons choisis aussi Docker car elle permet une facilité de portabilité de notre API et son installation.
Nous n'avons pas pris les alternatives comme Cononical LXD ou  Kubernetes car nous misosn sur nos conaissances préalable de Docker.
 
