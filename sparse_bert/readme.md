## 修改Transformers Library

- 在想要prune的module內部新增以下的member function
    
    ``` python
        def prune_embeddings(self, percentage):
            """
            Prune a percentage of the word embeddings and make the pruned parameters always zero.

            :param percentage: percentage of the embeddings to be pruned
            """
            import torch.nn.utils.prune as prune
            prune.l1_unstructured(self.word_embeddings, 'weight', amount=(percentage/100))

        def commit_sparsity_ratio(self):
            import torch.nn.utils.prune as prune
            prune.remove(self.word_embeddings, 'weight')
            total_elements = self.word_embeddings.weight.numel()
            zero_elements = (self.word_embeddings.weight == 0).sum().item()
            sparsity_ratio = zero_elements / total_elements
            return sparsity_ratio

    ```
    