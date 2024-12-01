# ***Baudel-AI.r***
**AUTHORS** : Clovis Lechien - clovis.lechien@epita.fr

**mk1/** : Première grosse version du projet, fine-tuning de gpt-2 medium, captionning simple avec BLIP-2, pipeline WandB, ...

**mk2/** : Seconde et version actuelle du projet, LLaVA (MLLM), gTTS (TTS).

## ***LLaVa*** : Multimodal Large Language Model (MLLM) 
### **Level 0** :
- [x] Le système avec l'ensemble de ses briques avec  une uniquement commande en local 
- [x] Les différents services du système sont dockerisés

### **Level 1** :
- [x] Le modèle est déployable automatiquement dans un environnement de production comme une VM déjà créée manuellement. Le déploiment se fait  soit via une commande simple soit via une pipeline CD. 
- [x] Le modèle n'est pas accessible sans clé “token” envoyé par l'utilisateur dans le header de la requête HTTP dans le cas d'un webservice. Un système de sécurité est mis en place dans les autres cas. Les tokens que le service accepte pourront être stocké en dur dans un .env ou dans un sqlite (ou autre mécanisme simple)

### **Level 2** : - Implémenter plusieurs options parmi les options suivantes : 

- [ ] La ou les VM sont déployées automatiquement avec du Terraform, du pulumi ou tout autre outils de déploiement de ressouces cloud
- [ ] Faire un test de charge de votre modèle en production avec un outil de type locust
- [ ] déployer vos conteneur avec kubernetes - POSSIBLE
- [x] avoir une interface via un bot discord ou une interface graphique pour utiliser votre systeme
- [x] Faire de la quantization de votre modèle pour booster ses performances
- [ ] faire du canary deployment
- [x] utiliser du MLFlow pour versionner son modèle et / ou mettre son modèle à jour en production
- [ ] faire du scaling horizontal
- [ ] dans le cas des LLM des système de protection contre le prompt injection sont mis en place - POSSIBLE
- [ ] Le modèle dispose d'un système de détection de data-drift 

## ***Instructions***
Head to **mk2/** and follow these steps:

How to use the service ?
```bash
docker-compose up --build
```
Wait for the docker containers to properly start.
Head up to **localhost:8051** to access the interface.

How to stop the service ?
```bash
docker-compose down
```
