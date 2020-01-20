'''
Moshe Binieli
'''
from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


K_PARAMETER = 10
EM_THRESHOLD = 5
LAMBDA_VALUE = 1.1
NUMBER_OF_CLUSTERS = 9
ALPHA_THRESHOLD = 0.000001


class Loader(object):
    def __init__(self, develop_set_file_name, topic_set_file_name):
        self.develop_set_file_name = develop_set_file_name
        self.topic_set_file_name = topic_set_file_name

    def get_topics(self):
        list_of_topics = []
        with open(self.topic_set_file_name) as file:
            for line in file:
                if line != '\n':
                    list_of_topics.append(line.strip())
        return list_of_topics

    def create_test_set(self):
        index_of_header = 0
        index_of_article = 0

        headers = {}  # Article headers.
        articles = {}  # Articles.
        word_frequency = {}

        with open(self.develop_set_file_name) as file:
            for line in file:
                row = line.strip().split(' ')
                indicator = row[0].split('\t')

                if indicator[0] == '':
                    continue

                if len(indicator) > 1:  # Header
                    headers[index_of_header] = row[0].replace("<", "").replace(">", "").split("\t")
                    index_of_header += 1
                else:  # Article
                    document_content = row
                    articles[index_of_article] = document_content
                    index_of_article += 1

                    for word in document_content:
                        if word not in word_frequency:
                            word_frequency.setdefault(word, 1)
                        else:
                            word_frequency[word] += 1

        word_frequency = self.delete_rare_words(word_frequency)
        articles = self.delete_rare_words_from_train(word_frequency, articles)
        article_frequencies = self.create_article_frequencies(articles)

        return headers, articles, word_frequency, article_frequencies

    def delete_rare_words(self, word_frequency):
        filtered_word_frequency_dictionary = {}

        for word, frequency in word_frequency.items():
            if frequency > 3:
                filtered_word_frequency_dictionary[word] = frequency

        return filtered_word_frequency_dictionary

    def delete_rare_words_from_train(self, word_frequency, articles):
        new_articles = {}

        for index, article_list in articles.items():
            article_content_after_del_rares = []
            for word in article_list:
                if word in word_frequency:
                    article_content_after_del_rares.append(word)
            new_articles[index] = article_content_after_del_rares

        return new_articles

    def create_article_frequencies(self, articles):
        count_each_article_words = {}

        for index, article_list in articles.items():
            count_each_article_words[index] = Counter(article_list)

        return count_each_article_words

    def divide_clusters(self, articles):
        clusters = {}

        for i in range(len(articles)):
            selected_cluster = (i+1) % NUMBER_OF_CLUSTERS

            if selected_cluster == 0:
                selected_cluster = NUMBER_OF_CLUSTERS

            if selected_cluster not in clusters:
                clusters[selected_cluster] = []

            clusters[selected_cluster].append(i)

        return clusters


class Utils(object):
    def add_tag_to_articles(self, clusters, articles):
        tagged_articles = []

        for cluster_id in articles:
            for x in articles[cluster_id]:
                tag = clusters[cluster_id]
                tagged_articles.append((x, tag))

        return tagged_articles

    def calculate_accuracy(self, headers, articles_by_topic):
        counter = 0

        for article in articles_by_topic:
            if article[1] in headers[article[0]]:
                counter += 1

        return counter / len(headers)

    def create_confusion_matrix(self, model_w, article_frequencies, topics, headers):
        articles_of_clusters = {}
        clusters_and_topics = {}

        topics_size = len(topics)
        clusters_size = len(model_w[0].keys())

        confusion_matrix = np.zeros((clusters_size, topics_size + 1))

        for article_cure in article_frequencies:
            index_of_max = 0
            max_w = model_w[article_cure][0]

            for i in range(clusters_size):
                if model_w[article_cure][i] > max_w:
                    max_w = model_w[article_cure][i]
                    index_of_max = i

            if index_of_max not in articles_of_clusters:
                articles_of_clusters[index_of_max] = []

            articles_of_clusters[index_of_max].append(article_cure)

        for x in range(clusters_size):
            for y in range(topics_size):
                cure_topic = topics[y]
                for t in articles_of_clusters[x]:
                    if cure_topic in headers[t]:
                        confusion_matrix[x][y] += 1

            confusion_matrix[x][topics_size] = len(articles_of_clusters[x])

        for x in range(clusters_size):
            most_topic_val = 0
            most_topic = 0

            for y in range(topics_size):
                if confusion_matrix[x][y] > most_topic_val:
                    most_topic = topics[y]
                    most_topic_val = confusion_matrix[x][y]
            clusters_and_topics[x] = most_topic

        return confusion_matrix, clusters_and_topics, articles_of_clusters


class EstimationMaximization(object):
    def run_algorithm(self, article_frequencies, word_frequency, articles_into_clusters, topics_size):
        likelihood_list = []
        perplexity_list = []

        iteration_index = 0
        current_likelihood = -10000000
        previous_likelihood = -20000000

        word_frequency_size = len(word_frequency)
        alpha_list, probabilities_list = self.init_probabilities_and_alpha(word_frequency, article_frequencies, articles_into_clusters, topics_size, word_frequency_size, LAMBDA_VALUE)

        word_frequency_values_size = sum(word_frequency.values())

        while current_likelihood - previous_likelihood > EM_THRESHOLD:
            w, z_values, max_zi_values = self.e_step(article_frequencies, alpha_list, probabilities_list, topics_size, K_PARAMETER)

            alpha_list, probabilities_list = self.m_step(w, article_frequencies, word_frequency, topics_size, LAMBDA_VALUE, word_frequency_size)
            previous_likelihood = current_likelihood

            current_likelihood = self.likelihood_calculation(max_zi_values, z_values, K_PARAMETER)
            print("Likelihood in iteration {} is {}.".format(iteration_index, current_likelihood))

            compute_current_perplexity = self.perplexity_calculation(current_likelihood, word_frequency_values_size)

            likelihood_list.append(current_likelihood)
            perplexity_list.append(compute_current_perplexity)
            iteration_index += 1

        self.plot_graph(iteration_index, likelihood_list, "Likelihood")
        self.plot_graph(iteration_index, perplexity_list, "Perplexity")

        return w

    def e_step(self, article_frequencies, alpha_list, probabilities_list, topics_size, k_parameter):
        max_zi_values = []
        z_values = []
        w = {}

        for article_index, frequencies in article_frequencies.items():
            w[article_index] = {}
            z_value_current_sum = 0

            # zi = ln(ai) + Sigma(Ntk) * Pik
            z_value, max_zi = self.compute_zs(topics_size, alpha_list, probabilities_list, frequencies)

            for i in range(topics_size):
                if z_value[i] - max_zi < -k_parameter:
                    w[article_index][i] = 0
                else:
                    w[article_index][i] = math.exp(z_value[i] - max_zi)
                    z_value_current_sum += w[article_index][i]

            for x in range(topics_size):
                w[article_index][x] /= z_value_current_sum

            z_values.append(z_value)
            max_zi_values.append(max_zi)

        return w, z_values, max_zi_values

    def m_step(self, w, article_frequencies, word_frequency, topics_size, lamda, word_frequency_size):
        number_articles = len(article_frequencies)
        alpha_list = [0] * topics_size
        probabilities = {}
        denominator_list = []

        # Sigma(Wti, Nt)
        for topic_index in range(topics_size):
            denominator = 0
            for i in article_frequencies:
                denominator += w[i][topic_index] * sum(article_frequencies[i].values())
            denominator_list.append(denominator)

        # Sigma(Wti, Ntk)
        for word in word_frequency:
            probabilities[word] = {}
            for topic_index in range(topics_size):
                numerator = 0
                for t in article_frequencies:
                    if word in article_frequencies[t] and w[t][topic_index] != 0:
                        numerator += w[t][topic_index] * article_frequencies[t][word]
                probabilities[word][topic_index] = self.lidstone_smooth(numerator, denominator_list[topic_index], word_frequency_size, lamda)

        for i in range(topics_size):
            for t in article_frequencies:
                alpha_list[i] += w[t][i]
            alpha_list[i] /= number_articles

        for i in range(len(alpha_list)):
            if alpha_list[i] < ALPHA_THRESHOLD:
                alpha_list[i] = ALPHA_THRESHOLD

        alfha_sum = sum(alpha_list)
        
        alpha_list = [topic_index / alfha_sum for topic_index in alpha_list]
        return alpha_list, probabilities

    def plot_graph(self, epochs, axis_y, label):
        axis_x = [i for i in range(epochs)]

        plt.plot(axis_x, axis_y, label=label)
        plt.xlabel("Epochs")
        plt.ylabel(label)
        plt.xlim(0, epochs)
        plt.ylim(min(axis_y), max(axis_y))
        plt.legend()
        plt.savefig(label + ".png")

    def init_probabilities_and_alpha(self, word_frequency, article_frequencies, articles_into_clusters, topics_size, word_frequency_size, lamda):
        weights = {}

        for clustered_index, clustered_articles_index in articles_into_clusters.items():
            for article_index in clustered_articles_index:
                weights[article_index] = {}
                weights[article_index][clustered_index-1] = 1
                for x in range(topics_size):
                    if x not in weights[article_index]:
                        weights[article_index][x] = 0

        return self.m_step(weights, article_frequencies, word_frequency, topics_size, lamda, word_frequency_size)

    def compute_zs(self, topics_size, alpha_list, probabilities_list, frequencies):
        # zi = ln(ai) + Sigma(Ntk) * Pik
        z_list = []

        for i in range(topics_size):
            sum_ln = 0
            for word in frequencies:
                sum_ln += np.log(probabilities_list[word][i]) * frequencies[word]
            z_to_add = np.log(alpha_list[i]) + sum_ln
            z_list.append(z_to_add)
 
        return z_list, max(z_list)

    def likelihood_calculation(self, max_zi_values, z_values, k_parameter):
        likelihood = 0

        for t in range(len(max_zi_values)):
            sum_zs_e = 0
            for i in range(len(z_values[t])):
                zi_m = z_values[t][i] - max_zi_values[t]
                if zi_m >= (-1.0) * k_parameter:
                    sum_zs_e += math.exp(zi_m)
            log_sum = np.log(sum_zs_e)
            likelihood += log_sum + max_zi_values[t]

        return likelihood

    def perplexity_calculation(self, current_likelihood, word_frequency_values_size):
        return math.pow(2, (-1 / word_frequency_values_size * current_likelihood))

    def lidstone_smooth(self, word_frequency, training_list_size, vocabulary_size, lamda):
        return (word_frequency + lamda) / (training_list_size + lamda * vocabulary_size)


def main():
    develop_set_file_name = "dataset/develop.txt"  # sys.argv[1]
    topic_set_file_name = "dataset/topics.txt"  # sys.argv[2]

    loader = Loader(develop_set_file_name, topic_set_file_name)
    topics = loader.get_topics()
    headers, articles, word_frequency, article_frequencies = loader.create_test_set()
    articles_into_clusters = loader.divide_clusters(articles)

    em = EstimationMaximization()
    w_model = em.run_algorithm(article_frequencies, word_frequency, articles_into_clusters, len(topics))

    utils = Utils()
    confusion_matrix, clusters_and_topics, articles_of_clusters = utils.create_confusion_matrix(w_model, article_frequencies, topics, headers)
    print(confusion_matrix)
    
    articles_by_topic = utils.add_tag_to_articles(clusters_and_topics, articles_of_clusters)
    print("Model accuracy {}.".format(round(utils.calculate_accuracy(headers, articles_by_topic), 6)))


if __name__ == "__main__":
    main()
