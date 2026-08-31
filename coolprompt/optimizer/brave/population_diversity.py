from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from sentence_transformers import SentenceTransformer

from coolprompt.optimizer.reflective_prompt.prompt import Prompt


class BERTSimilarityComputer:
    """Compute semantic similarity using BERT embeddings"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """Initialize a model for semantic similarity.

        Args:
            model_name (str): SentenceTransformers model identifier.
        """
        self.model = SentenceTransformer(model_name)
        self.available = True

    def compute_similarity(self, texts: List[str]) -> np.ndarray:
        """Compute a semantic similarity matrix using BERT.

        Args:
            texts (List[str]): texts to compare.

        Returns:
            np.ndarray: pairwise cosine similarities, or ``None`` when fewer
                than two texts are supplied or encoding fails.
        """
        if not self.available or len(texts) < 2:
            return None

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            similarity = cosine_similarity(embeddings)
            return similarity
        except Exception:
            return None


class PopulationDiversityManager:
    """Manages population diversity by removing similar prompts"""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_per_cluster: int = 2,
        auto_threshold: bool = True,
        target_cluster_count: int = None,
        use_hierarchical: bool = True,
        use_bert: bool = True,
        bert_weight: float = 0.6,
        duplicate_threshold: float = 0.95
    ):
        """Initialize population-diversity controls.

        Args:
            similarity_threshold (float): cosine-similarity clustering cutoff.
            max_per_cluster (int): maximum prompts retained per cluster.
            auto_threshold (bool): whether to adapt the clustering threshold.
            target_cluster_count (int): desired number of clusters.
            use_hierarchical (bool): whether to use hierarchical clustering.
            use_bert (bool): whether to blend semantic BERT similarity.
            bert_weight (float): BERT weight in the hybrid similarity.
            duplicate_threshold (float): cutoff for near-duplicate removal.
        """
        self.similarity_threshold = similarity_threshold
        self.max_per_cluster = max_per_cluster
        self.auto_threshold = auto_threshold
        self.target_cluster_count = target_cluster_count
        self.use_hierarchical = use_hierarchical
        self.use_bert = use_bert
        self.bert_weight = bert_weight
        self.duplicate_threshold = duplicate_threshold

        # TF-IDF for fast syntactic similarity
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            lowercase=True,
            max_features=500,
            min_df=1
        )

        # BERT for semantic similarity
        self.bert_computer = None
        if use_bert:
            self.bert_computer = BERTSimilarityComputer()

        self.last_filter_report = {}

    def _compute_similarity_matrix(self, prompts: List[Prompt]) -> np.ndarray:
        """Compute a hybrid TF-IDF and BERT similarity matrix.

        Args:
            prompts (List[Prompt]): prompts to compare.

        Returns:
            np.ndarray: square pairwise-similarity matrix.
        """
        texts = [p.text for p in prompts]

        if len(texts) < 2:
            return np.array([[1.0]])

        # Compute TF-IDF similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            tfidf_sim = cosine_similarity(tfidf_matrix)
        except ValueError:
            tfidf_sim = np.ones((len(texts), len(texts)))

        # Compute BERT similarity if available
        if self.use_bert and self.bert_computer is not None and \
                self.bert_computer.available:
            bert_sim = self.bert_computer.compute_similarity(texts)
            if bert_sim is not None:
                # Hybrid similarity: weighted average
                similarity = (
                    (1 - self.bert_weight) * tfidf_sim +
                    self.bert_weight * bert_sim
                )
                return similarity

        return tfidf_sim

    def _adaptive_threshold(
        self,
        similarity_matrix: np.ndarray,
        population_size: int
    ) -> float:
        """Choose a threshold near the target number of clusters.

        Args:
            similarity_matrix (np.ndarray): pairwise similarities.
            population_size (int): number of prompts represented by the matrix.

        Returns:
            float: adapted threshold, or the configured fixed threshold.
        """
        if population_size < 2 or self.target_cluster_count is None:
            return self.similarity_threshold

        # For hierarchical clustering
        if self.use_hierarchical:
            distance_matrix = 1 - similarity_matrix
            # Make distance matrix for clustering
            upper_triangle = distance_matrix[
                np.triu_indices_from(distance_matrix, k=1)
            ]
            if len(upper_triangle) == 0:
                return self.similarity_threshold

            # Binary search for threshold
            low, high = 0.0, 1.0
            best_threshold = self.similarity_threshold

            for _ in range(15):
                mid = (low + high) / 2
                # Convert similarity threshold to distance threshold
                distance_threshold = 1 - mid

                try:
                    if len(upper_triangle) > 0:
                        Z = linkage(upper_triangle, method='complete')
                        clusters = fcluster(
                            Z,
                            distance_threshold,
                            criterion='distance'
                        )
                        num_clusters = len(np.unique(clusters))
                    else:
                        num_clusters = len(similarity_matrix)
                except Exception:
                    num_clusters = len(similarity_matrix)

                if num_clusters < self.target_cluster_count:
                    high = mid  # Need more clusters, raise threshold
                else:
                    low = mid   # Have enough clusters, lower threshold

                if abs(num_clusters - self.target_cluster_count) <= 1:
                    best_threshold = mid
                    break

            return best_threshold
        else:
            return self.similarity_threshold

    def _hierarchical_clustering(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> List[List[int]]:
        """Cluster prompts with complete-linkage hierarchical clustering.

        Args:
            similarity_matrix (np.ndarray): pairwise similarities.
            threshold (float): minimum within-cluster similarity.

        Returns:
            List[List[int]]: clusters of prompt indices.
        """
        distance_matrix = 1 - similarity_matrix
        upper_triangle = distance_matrix[
            np.triu_indices_from(distance_matrix, k=1)
        ]

        if len(upper_triangle) == 0:
            return [[i] for i in range(len(similarity_matrix))]

        try:
            Z = linkage(upper_triangle, method='complete')
            distance_threshold = 1 - threshold
            clusters_arr = fcluster(Z, distance_threshold, criterion='distance')

            clusters = {}
            for idx, cluster_id in enumerate(clusters_arr):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(idx)

            return list(clusters.values())
        except Exception:
            return [[i] for i in range(len(similarity_matrix))]

    def _dfs_clustering(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> List[List[int]]:
        """Cluster prompts as connected components using DFS.

        Args:
            similarity_matrix (np.ndarray): pairwise similarities.
            threshold (float): minimum edge similarity.

        Returns:
            List[List[int]]: clusters of prompt indices.
        """
        n = len(similarity_matrix)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            cluster = []
            stack = [i]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                cluster.append(node)

                for j in range(n):
                    if not visited[j] and \
                       similarity_matrix[node][j] >= threshold:
                        stack.append(j)

            clusters.append(sorted(cluster))

        return clusters

    def _filter_near_duplicates(
        self,
        population: List[Prompt],
        similarity_matrix: np.ndarray,
        duplicate_threshold: float = 0.95
    ) -> Tuple[List[Prompt], np.ndarray, np.ndarray]:
        """Remove near-duplicates, keeping the best of each group.

        Args:
            population (List[Prompt]): prompts sorted by descending score.
            similarity_matrix (np.ndarray): pairwise similarities.
            duplicate_threshold (float): duplicate-similarity cutoff.

        Returns:
            Tuple[List[Prompt], np.ndarray, np.ndarray]: retained prompts,
                reduced similarity matrix, and retained original indices.
        """
        n = len(population)
        visited = [False] * n
        kept_indices = []

        for i in range(n):
            if visited[i]:
                continue

            duplicates = [i]
            for j in range(i + 1, n):
                if not visited[j] and \
                   similarity_matrix[i][j] >= duplicate_threshold:
                    duplicates.append(j)
                    visited[j] = True

            # Keep only best (first in sorted list)
            best_idx = duplicates[0]
            kept_indices.append(best_idx)
            visited[best_idx] = True

        # Create new population and similarity matrix
        kept_population = [population[i] for i in kept_indices]
        kept_sim_matrix = similarity_matrix[np.ix_(kept_indices, kept_indices)]

        return kept_population, kept_sim_matrix, kept_indices

    def filter_by_diversity(
        self,
        population: List[Prompt],
        target_population_size: int
    ) -> List[Prompt]:
        """Filter a population for diversity while retaining strong prompts.

        Args:
            population (List[Prompt]): prompts sorted by descending score.
            target_population_size (int): maximum returned population size.

        Returns:
            List[Prompt]: diverse population sorted by descending score.
        """
        if len(population) <= target_population_size:
            return population

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(population)

        # Level 1: Remove near-duplicates (threshold similarity)
        (
            population,
            similarity_matrix,
            kept_indices
        ) = self._filter_near_duplicates(
            population,
            similarity_matrix,
            duplicate_threshold=self.duplicate_threshold
        )

        if len(population) <= target_population_size:
            return sorted(population, key=lambda p: p.score, reverse=True)

        # Determine threshold
        if self.auto_threshold:
            threshold = self._adaptive_threshold(
                similarity_matrix,
                target_population_size
            )
        else:
            threshold = self.similarity_threshold

        threshold = max(
            self.similarity_threshold,
            min(threshold, self.duplicate_threshold)
        )

        # Cluster prompts
        if self.use_hierarchical:
            clusters = self._hierarchical_clustering(
                similarity_matrix,
                threshold
            )
        else:
            clusters = self._dfs_clustering(similarity_matrix, threshold)

        # Select best prompts from each cluster
        selected_indices = []
        removed_indices = []

        for cluster in clusters:
            # Keep top max_per_cluster from this cluster
            # (already sorted by score)
            for idx in cluster[:self.max_per_cluster]:
                selected_indices.append(idx)

            # Track removed prompts
            for idx in cluster[self.max_per_cluster:]:
                removed_indices.append(idx)

        # Store report for logging
        self.last_filter_report = {
            'threshold': threshold,
            'num_clusters': len(clusters),
            'num_removed': len(removed_indices),
            'removed_indices': removed_indices,
            'similarity_matrix': similarity_matrix,
            'deduplication_removed': len(kept_indices)
        }

        # Sort by original order and select
        selected_indices = sorted(selected_indices)
        selected_prompts = [population[i] for i in selected_indices]

        # Re-sort by score and trim to target size
        selected_prompts = sorted(
            selected_prompts,
            key=lambda p: p.score,
            reverse=True
        )[:target_population_size]

        return selected_prompts

    def maintain_diversity(
        self,
        population: List[Prompt],
        max_size: int
    ) -> List[Prompt]:
        """Sort prompts by score and enforce a diverse population bound.

        Args:
            population (List[Prompt]): candidate prompts.
            max_size (int): maximum returned population size.

        Returns:
            List[Prompt]: diverse population sorted by descending score.
        """
        if len(population) <= max_size:
            return sorted(population, key=lambda p: p.score, reverse=True)

        # First sort by score
        sorted_pop = sorted(population, key=lambda p: p.score, reverse=True)

        # Then filter by diversity
        diverse_pop = self.filter_by_diversity(sorted_pop, max_size)

        # Final sort by score
        return sorted(diverse_pop, key=lambda p: p.score, reverse=True)

    def compute_diversity(self, population: List[Prompt]) -> float:
        """Return mean pairwise distance in [0, 1] using TF-IDF only (fast).

        Returns 1.0 (fully diverse) when population has fewer than 2 prompts.

        Args:
            population (List[Prompt]): prompts whose diversity to measure.

        Returns:
            float: mean pairwise distance, or ``0.5`` if computation fails.
        """
        if len(population) < 2:
            return 1.0
        texts = [p.text for p in population]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            sim = cosine_similarity(tfidf_matrix)
            n = len(population)
            upper = sim[np.triu_indices(n, k=1)]
            return float(np.clip(1.0 - upper.mean(), 0.0, 1.0))
        except Exception:
            return 0.5

    def get_filter_report(self) -> dict:
        """Get details about the last filtering operation.

        Returns:
            dict: thresholds, clusters, and removed prompt indices.
        """
        return self.last_filter_report
