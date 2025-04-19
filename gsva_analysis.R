# ===== Load Libraries =====
library(GSVA)
library(GSEABase)
library(Biobase)
library(msigdbr)

# ===== Load Expression Matrix (already HGNC mapped and log-normalized) =====
expr <- read.csv("dataset/ccle_expression_mapped_log.csv", row.names = 1)

# Clean and sanitize the matrix
expr_matrix <- as.matrix(expr)
storage.mode(expr_matrix) <- "double"
expr_matrix <- expr_matrix[complete.cases(expr_matrix), ]
expr_matrix <- expr_matrix[, colSums(is.na(expr_matrix)) == 0]

# ===== Load KEGG Medicus Gene Sets =====
kegg_sets_df <- msigdbr(species = "Homo sapiens", collection = "C2", subcollection = "CP:KEGG_MEDICUS")

# Prepare gene set list
gene_sets <- split(kegg_sets_df$gene_symbol, kegg_sets_df$gs_name)
gene_sets <- lapply(gene_sets, unique)

# Convert to GeneSetCollection
gene_sets_gmt <- GeneSetCollection(mapply(function(genes, name) {
  GeneSet(setName = name, geneIds = genes)
}, gene_sets, names(gene_sets)))

# ===== Run GSVA =====
unlockBinding("rowVars", asNamespace("matrixStats"))

assign("rowVars", function(x, rows = NULL, cols = NULL, na.rm = FALSE,
                           center = NULL, ...) {
  matrixStats::colVars(t(x), rows = cols, cols = rows, na.rm = na.rm,
                       center = if (is.null(center)) NULL else t(center),
                       useNames = FALSE)  # <-- THIS is the fix
}, envir = asNamespace("matrixStats"))

lockBinding("rowVars", asNamespace("matrixStats"))

gsva_results <- gsva(expr_matrix, gene_sets_gmt, method = "gsva")

# ===== Save Output =====
write.csv(gsva_results, "dataset/ccle_gsva_scores.csv")
