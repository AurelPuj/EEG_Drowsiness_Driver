<h2> installation </h2>
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
