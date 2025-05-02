import pika, json, time, os, ssl

import pika.exceptions
from pymongo import MongoClient
from google.cloud import secretmanager
from openai import OpenAI
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging


# Configura il logging all'inizio del file (se non già configurato altrove)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#

class AIProcessor:
    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_queue: str,
        rabbitmq_port: int,
        rabbitmq_username: str,
        rabbitmq_password: str,
        mongo_uri: str,
        mongo_database: str,
        mysql_uri : str,
        ollama_base_url: str = "http://192.168.1.187:11434/v1",
        ollama_api_key: str = "ollama",
        ai_model: str = "qwen2.5:7b"
    ):
        """
        Inizializza il processore AI con le configurazioni per RabbitMQ, MongoDB e Ollama.
        
        Args:
            rabbitmq_host: Hostname del server RabbitMQ
            rabbitmq_queue: Nome della coda RabbitMQ da cui consumare i messaggi
            mongo_uri: URI di connessione a MongoDB
            mongo_database: Nome del database MongoDB
            mongo_collection: Nome della collezione MongoDB
            ollama_base_url: URL base per l'API Ollama
            ollama_api_key: Chiave API per Ollama (default "ollama")
            ai_model: Modello AI da utilizzare (default "gemma:12b-instruct")
        """
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_queue = rabbitmq_queue
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_username = rabbitmq_username
        self.rabbitmq_password = rabbitmq_password
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_database
        self.ai_model = ai_model
        self.mysql_uri = mysql_uri

        # Inizializza il client OpenAI per Ollama
        self.ollama_client = OpenAI(
            base_url=ollama_base_url,
            api_key=ollama_api_key,
            timeout=None
        )
        
        # Connessioni verranno stabilite quando necessario
        self.rabbit_connection = None
        self.mongo_client = None

        self.mysql_engine = None
        self.mysql_session = None
    
    def connect_to_mongodb(self):
        """Stabilisce la connessione a MongoDB."""
        self.mongo_client = MongoClient(self.mongo_uri)
        self.db = self.mongo_client[self.mongo_db]
        self.collection = self.db["rss-feed-items"]
    
    def connect_to_mysql(self):
        """Stabilisce la connessione a MySQL."""
        if not self.mysql_engine:
            self.mysql_engine = create_engine(self.mysql_uri)
            Session = sessionmaker(bind=self.mysql_engine)
            self.mysql_session = Session()
        
    def connect_to_rabbitmq(self):

        """Stabilisce la connessione a RabbitMQ."""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.set_ciphers('ECDHE+AESGCM:!ECDSA')

        # url = f"amqps://autoblog-user:autoblog-user!@localhost:15671"
        url = f"amqps://{self.rabbitmq_username}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}"
        print(url)
        parameters = pika.URLParameters(url)
        parameters.ssl_options = pika.SSLOptions(context=ssl_context)

        self.rabbit_connection = pika.BlockingConnection(parameters)
        self.channel = self.rabbit_connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.channel.queue_declare(queue=self.rabbitmq_queue, durable=True)

    def get_document_by_link(self, link: str, processed=False) -> Optional[Dict[str, Any]]:
        """
        Cerca un documento in MongoDB usando il campo LINK.
        
        Args:
            link: Valore del campo LINK da cercare
            
        Returns:
            Il documento trovato o None se non esiste
        """
        if not self.mongo_client:
            self.connect_to_mongodb()
        
        elem = self.collection.find_one({"link": link})

        if elem:
            self.collection.update_one({"link": link}, {"$set" : {"processed" : processed}})

        return elem
    
    def generate_post(self, prompt: str, link: str) -> dict:
        """
        Invia un prompt a gemma:12b-instruct usando l'API compatibile OpenAI di Ollama.
        
        
        Args:
            prompt: Il prompt da inviare all'AI
            
        Returns:
            La risposta generata dall'AI
        """
        
        import html2text
        
        prompt = html2text.html2text(prompt, bodywidth=0)
         
        body = self.ollama_client.chat.completions.create(
            model=self.ai_model,
            timeout=60*3,
            messages=[
                {
                    "role": "system",
                    "content": """Sei un copywriter esperto nella creazione di contenuti per blog in lingua italiana. Ogni volta che ricevi un input, il tuo compito è scrivere un post strutturato come segue:

- Inizia con una frase coinvolgente che catturi l'attenzione del lettore, stimolando la curiosità.
- Approfondisci il tema in più paragrafi o sotto-sezioni. Assicurati di fornire informazioni utili e dettagliate, citando sempre le fonti quando fai riferimento a fatti, dati o articoli specifici.
- Concludi il post invitando il lettore a commentare, condividere il contenuto o approfondire l'argomento.
- Alla fine del post introduci il link di riferimento al feed: 'Riferimento: [link]({})' e saluta tutti con uno '\nGrazie per la vostra attenzione, stay updated!'

Termina con \"Stay updated!\"""".format(link),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6
        ).choices[0].message.content

        
        title = self.ollama_client.chat.completions.create(
            model=self.ai_model,
            timeout=60*3,
            messages=[
                {
                    "role": "system",
                    "content": """Sei un copywriter esperto in titolazione SEO.  
Il tuo compito è creare un titolo in italiano per l’articolo che riceverai.  
– Mantieni in inglese i termini tecnici (es. cloud, hybrid).  
– Lunghezza massima: 190 caratteri.  
– Restituisci **solo** il titolo, senza virgolette, punti elenco o altre formattazioni.""",
                },
                {"role": "user", "content": body},
            ],
            temperature=0.6
        ).choices[0].message.content

        tags = self.ollama_client.chat.completions.create(
            model=self.ai_model,
            timeout=60*3,
            messages=[
                {
                    "role": "system",
                    "content": """Sei un copywriter esperto in SEO e content marketing. Riceverai un articolo e il tuo compito è estrarne fino a 5 tag rilevanti, per ottimizzarne la discovery.
                    - Restituisci solo i tag, in una lista separata da virgole.  
                    - Non aggiungere numeri, virgolette, né testo descrittivo.""",
                },
                {"role": "user", "content": body},
            ],
            temperature=0.6
        ).choices[0].message.content

        
        categories = self.mysql_session.execute(text("SELECT * FROM categories")).fetchall()
        categories = "\n".join(["{} => {}".format(k, v) for k, v in categories])
        category = self.ollama_client.chat.completions.create(
            model=self.ai_model,
            timeout=60*3,
            messages=[
                {
                    "role": "system",
                    "content": "Sei un copywriter. Il tuo compito è selezionare una sola categoria leggendo l'articolo fornito. Le categorie disponibili sono “{}”. Rispondi esclusivamente con l’ID numerico (ad esempio 5 oppure 6), senza virgolette, punti, testo aggiuntivo, né spiegazioni. In caso non sai a quale categoria assegnarlo imposta solo 'None'.".format(categories),
                },
                {"role": "user", "content": body},
            ],
            temperature=0.6
        ).choices[0].message.content

        return {
            "title" : title,
            "body" : body,
            "tags" : [tag.strip() for tag in tags.split(",")],
            "category" : category 
        }
    
    def process_message(self, ch, method, properties, body):
        """
        Callback per elaborare i messaggi ricevuti da RabbitMQ.
        
        Args:

            ch: Canale RabbitMQ
            method: Metodo di consegna
            properties: Proprietà del messaggio
            body: Corpo del messaggio (contenente il LINK)
        """
        try:
            message = json.loads(body)
            link = message.get("link")
            
            if not link:
                logger.error("Errore: Il messaggio non contiene il campo LINK")
                return
            
            print(f"Ricerca documento con LINK: {link}")
            document = self.get_document_by_link(link)
            
            if not document:
                logger.warning(f"Documento con LINK {link} non trovato")
                return
            
            logger.info("Documento trovato, elaborazione con AI...")

            post = message.get('content', '')
            ai_response = self.generate_post(post, link=document["link"])

            logger.info("Risposta ricevuta: {}".format(json.dumps(ai_response)))

            # ai_response = {'title': 'Implementazione di una governance finanziaria per modelli cloud multicloud e hybrid: un esempio con Azure, AWS e GCP', 'body': "### Introduzione\n\nCari lettori,\n\nOggi ci troviamo in un mondo dove le organizzazioni e le imprese si trovano a gestire risorse cloud pluridirezionali, compresi modelli di nuvola multicloud e hybrid. In questo articolo esploriamo come implementare una governance finanziaria per queste configurazioni cloud, focalizzandoci in particolare su Microsoft Azure, Amazon Web Services (AWS) e Google Cloud Platform (GCP). Scopriremo come monitorare costi e bilanciare spese in un quadro uniforme, consentendo un'ottimizzazione costiera che include tutte le cariche di lavoro, sia in locale, in cloud o in ambiente misto.\n\n### La Necessità di una Governance Finanziaria\n\nIniziamo con un breve esempio: l’organizzazione DevOps ABCs ha un footprint cloud distribuito tra Azure (7 sottoscrizioni attive), GCP (2 conti fatturati) e AWS (un singolo account). Dalla data 1 Settembre 2018, la loro infrastruttura include due server fisici: Windows Server A con Microsoft SQL Server e Linux Server A con MySQL. Entrambi sono ospitati in loco.\n\n### Implementazione del Dashboard di Costi\n\n#### Azure\n\nIn Azure, organizziamo le risorse utilizzando Gruppi di Gestione (Management Groups) per garantire che risorse dipendenti con lo stesso ciclo di vita siano contenute nello stesso gruppo. Ad esempio, un'app web con un backend di database in un ambiente specifico (Dev, Test o Prod) dovrebbe essere inclusa nello stesso gruppo di risorse; se sono utilizzati tutti i tre ambienti, ne sarebbero necessari tre gruppi separati.\n\nPer monitorare le spese in Azure e AWS, utilizziamo il Cost Management in Azure per integrare dati da diversi provider. Questo implica l'utilizzo di API come [Azure Cost Management APIs](https://docs.microsoft.com/en-us/rest/api/cost-management/) e Data Factory per la raccolta dei dati.\n\n#### AWS\n\nIn AWS, utilizziamo il Cost Management in Azure per monitorare le spese. Ciò è possibile configurando l'integrazione tra Cost and Usage Reports di AWS e Azure Cost Management. Questo consente di importare i dati delle spese direttamente nel dashboard di Azure.\n\n### Approfondimenti\n\nLa governance finanziaria in cloud richiede un'attenzione speciale per quanto riguarda la stima dei costi. Le principali sfide includono:\n\n1. **Stima Costosa e Precisa**: Per le organizzazioni con infrastrutture miste, è cruciale avere una stima delle spese iniziali che sia precisa e affidabile.\n2. **Tagging Strategico**: La strategia di etichettamento corretta può facilitare l'attribuzione dei costi a progetti specifici o centri di costo.\n3. **Consolidamento dei Costi**: Utilizzare strumenti come Power BI per creare un dashboard interattivo che riunisca dati da diversi provider cloud.\n\n### Conclusione\n\nRaccogliere e monitorare i costi in tempo reale non solo permette di ottimizzare le spese, ma consente anche una maggiore trasparenza finanziaria. Questo è particolarmente importante per organizzazioni che utilizzano modelli cloud multicloud o hybrid.\n\n### Stay Update!\n\nPer mantenerti aggiornato su nuove evoluzioni e miglioramenti nella governance finanziaria cloud, sii attento alle novità nel [Microsoft Premier Developer Blog](https://devblogs.microsoft.com/premier-developer/cloud-economics/) e visita la nostra sezione di approfondimenti per ottenere ulteriori risorse.\n\nSegui il nostro blog per continuare a scoprire come ottimizzare le tue spese cloud in modo efficiente.", 'tags': ['tag', 'cloud', 'governance_finanziaria', 'azure', 'aws', 'gcp', 'costi_cloud', 'dashboards', 'api', 'etichettamento', 'stima_costi', 'trasparenza_financeira', 'hybrid_cloud', 'multicloud'], 'category': '5'} 

            if 'none' in ai_response["category"].lower():
                ai_response["category"] = None

            self.mysql_session.execute(
                text("""
                    INSERT INTO posts
                    (title, body, image, publisher_id, category_id, status_id)
                    VALUES(:title, :body, '/static/images/default.png', NULL, :category, 1);
                """),
                ai_response
            )

            self.mysql_session.commit()

            result = self.mysql_session.execute(text("SELECT LAST_INSERT_ID()"))
            post_id = result.scalar()  # Ottieni l'ID

            for tag in ai_response["tags"]:
                tag_sql = self.mysql_session.execute(text("SELECT * FROM tags WHERE tag = :tag"), {"tag" : tag}).fetchone()

                if not tag_sql:
                    self.mysql_session.execute(
                        text("""INSERT INTO tags(tag) VALUES(:tag);"""),
                        {"tag" : tag}
                    )
                    self.mysql_session.commit()

                    tag_id = self.mysql_session.execute(text("SELECT LAST_INSERT_ID()"))
                    tag_id = tag_id.scalar()  # Ottieni l'ID
                else:
                    tag_id = tag_sql[0]

                self.mysql_session.execute(
                    text("""INSERT INTO post_tags VALUES (:post_id, :tag_sql) ON DUPLICATE KEY UPDATE tag_id = VALUES(tag_id)"""),
                    {
                        "post_id" : post_id,
                        "tag_sql" : tag_id
                    }
                )
                self.mysql_session.commit()
                self.get_document_by_link(link, processed=True)
        except Exception as e:
            logger.error(f"Errore durante l'elaborazione del messaggio: {e}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_consuming(self):
        """Avvia il consumo dei messaggi dalla coda RabbitMQ con gestione disconnessione."""
        while True:
            try:
                if not self.rabbit_connection or self.rabbit_connection.is_closed:
                    self.connect_to_rabbitmq()

                if not self.mysql_engine:
                    self.connect_to_mysql()

                self.channel.basic_qos(prefetch_count=1)

                print(f"In attesa di messaggi dalla coda {self.rabbitmq_queue}...")
                self.channel.basic_consume(
                    queue=self.rabbitmq_queue,
                    on_message_callback=self.process_message,
                    auto_ack=False,
                )

                self.channel.start_consuming()

            except KeyboardInterrupt:
                print("Interruzione manuale. Chiudo connessioni...")
                if self.channel.is_open:
                    self.channel.stop_consuming()
                if self.rabbit_connection and not self.rabbit_connection.is_closed:
                    self.rabbit_connection.close()
                if self.mongo_client:
                    self.mongo_client.close()
                break

            except (pika.exceptions.StreamLostError, pika.exceptions.AMQPConnectionError) as e:
                print(f"[!] Connessione RabbitMQ persa: {e}. Riprovo tra 5 secondi...")
                time.sleep(5)


def access_secret(secret_id, project_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Esempio di utilizzo
if __name__ == "__main__":
    gcp_project_id = os.environ.get("GCP_PROJECT_ID", "gcp-automated-blog-test")
    rabbit_mq_conn_secret = os.environ.get("GCP_RABBIT_MQ_SECRET", "rabbit-connection")
    mongodb_conn_secret = os.environ.get("GCP_MONGODB_SECRET", "mongodb-connection")
    mysql_conn_secret = os.environ.get("GCP_MYSQL_SECRET", "mysql-connection")
    
    # Recupero i segreti da Secret Manager
    rabbit_mq_credentials = json.loads(access_secret(rabbit_mq_conn_secret, gcp_project_id))
    mongodb_credentials = json.loads(access_secret(mongodb_conn_secret, gcp_project_id))
    mysql_conn_secret = json.loads(access_secret(mysql_conn_secret, gcp_project_id))


    processor = AIProcessor(
        **rabbit_mq_credentials,
        **mongodb_credentials,
        **mysql_conn_secret,
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.187:11434/v1")
    )
    
    processor.start_consuming()