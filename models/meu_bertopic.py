from bertopic import BERTopic as BERTopic_
import numpy as np
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt

class BERTopic(BERTopic_):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def print_params(self):
        print("Parâmetros do modelo:")
        # Acessando os atributos da classe pai
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def save_txt(self, pathfile: str) -> None:
        with open(pathfile, 'w') as f:
            for i in range(len(self.get_topic_info())):
                t = self.get_topic_info()['Topic'][i]
                f.write(f'\ntopico {t}:\n')
                words = self.get_topic_info()['Representation'][i]
                for word in words:
                    f.write(f'{word} ')

    def save_json(self, pathfile: str) -> None:
        topics = self.get_topics()
        result = {}
        for topic_id, words in topics.items():
            result[topic_id] = [[word, value] for word, value in words if word != ""]

        with open(pathfile, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

    def dominant_topics(self, data: list, path: str, ids: list) -> None:

        topicnames = ['Topico ' + str(i) for i in self.get_topic_info()["Topic"].values.tolist()]
        papernames = [str(i) for i in ids]
        topic_dist, _ = self.approximate_distribution(data)
        if "Topico -1" in topicnames:
            temp_array = 1 - topic_dist.sum(axis=1)
            topic_dist = np.insert(topic_dist, 0, temp_array, axis=1)

        df_document_topic = pd.DataFrame(np.round(topic_dist, 4), columns=topicnames)
        df_document_topic['id'] = papernames
        df_document_topic['dominant_topic'] = self.topics_

        sns.countplot(x=df_document_topic.dominant_topic)
        plt.savefig(f'./topicmodeling/{path}/Topicos_Dominantes.png')
        plt.close()

        df_document_topic.to_csv(f'./topicmodeling/{path}/Topicos_Dominantes.csv', sep="|")
        resumo = pd.DataFrame()
        resumo['papers'] = papernames
        resumo['dominant_topic'] = df_document_topic['dominant_topic'].values
        resumo.to_csv(f'./topicmodeling/{path}/Resumo_Topicos_Dominantes.csv', index=False)
