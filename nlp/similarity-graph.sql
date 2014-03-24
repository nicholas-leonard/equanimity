--DROP TABLE bw.sentence_bag;
CREATE TABLE bw.sentence_bag (
	sentence_id	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
CREATE INDEX sentence_bag_sentence_id ON bw.sentence_bag (sentence_id);
CREATE INDEX sentence_bag_word_id ON bw.sentence_bag (word_id);

--tf
INSERT INTO bw.sentence_bag (sentence_id, word_id, word_freq) (
	SELECT sentence_id, word_id, COUNT(*)
	FROM	(
		SELECT sentence_id, unnest(sentence_words) AS word_id 
		FROM bw.train_sentence
		WHERE sentence_id BETWEEN 1 AND 1111111
		) AS a
	GROUP BY sentence_id, word_id
);--27340529 rows affected

--idf : http://en.wikipedia.org/wiki/Tf%E2%80%93idf
--DROP TABLE bw.word_idf;
CREATE TABLE bw.word_idf (
	word_id		INT4,
	word_idf	FLOAT8,
	PRIMARY KEY (word_id)
);

CREATE OR REPLACE FUNCTION bw.select_idf( word_id INT4) RETURNS FLOAT8 AS $$
	SELECT COUNT(*)::FLOAT8 FROM bw.sentence_bag AS b WHERE b.word_id = $1
$$ LANGUAGE 'SQL';

INSERT INTO bw.word_idf (word_id, word_idf) (
	SELECT word_id, log(1000000/bw.select_idf(a.word_id))
	FROM 	(
		SELECT DISTINCT word_id 
		FROM bw.sentence_bag
		) AS a
);--317931 rows affected

--tf-idf
UPDATE bw.sentence_bag AS a
SET word_freq = word_freq * word_idf
FROM bw.word_idf AS b
WHERE a.word_id = b.word_id--24607658 rows affected

--optimize
--DROP TABLE bw.stop_word;
CREATE TABLE bw.stop_word (
	word_id		INT4,
	PRIMARY KEY (word_id)
);
INSERT INTO bw.stop_word (word_id) (
	SELECT word_id 
	FROM bw.word_count 
	ORDER BY word_count DESC LIMIT 100
);

--DROP TABLE bw.sentence_bag_s;
CREATE TABLE bw.sentence_bag_s (
	sentence_id	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.sentence_bag_s (sentence_id, word_id, word_freq) (
	SELECT sentence_id, word_id, word_freq
	FROM bw.sentence_bag AS a
	WHERE (SELECT b.word_id FROM bw.stop_word AS b WHERE b.word_id = a.word_id LIMIT 1) IS NULL
	ORDER BY sentence_id
);
CREATE INDEX sentence_bag_s_idx ON bw.sentence_bag_s (sentence_id);

--DROP TABLE bw.sentence_bag_w;
CREATE TABLE bw.sentence_bag_w (
	sentence_id	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.sentence_bag_w (sentence_id, word_id, word_freq) (
	SELECT sentence_id, word_id, word_freq
	FROM bw.sentence_bag AS a
	WHERE (SELECT b.word_id FROM bw.stop_word AS b WHERE b.word_id = a.word_id LIMIT 1) IS NULL
	ORDER BY word_id
);
CREATE INDEX sentence_bag_w_idx ON bw.sentence_bag_w (word_id);


--DROP TABLE bw.sentence_similarity;
CREATE TABLE bw.sentence_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	

--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_sentence_similarity_graph ( sentence_id INT4 ) RETURNS VOID AS $$
INSERT INTO bw.sentence_similarity (tail, head, similarity) (
        SELECT  $1,
                sentence_id,
                dot_product
                /(
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.sentence_bag_s AS c
                WHERE c.sentence_id = a.sentence_id
                )
                *
                b.norm
                ) AS similarity                
        FROM    (
                SELECT sentence_id, dot_product
                FROM    (
                        SELECT b.sentence_id, SUM(a.word_freq*b.word_freq)::FLOAT8 AS dot_product
                        FROM bw.sentence_bag_s AS a, bw.sentence_bag_w AS b
                        WHERE a.sentence_id = $1 AND a.word_id = b.word_id 
                        GROUP BY b.sentence_id
                        ) AS a
                WHERE sentence_id != $1
                ) AS a,
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.sentence_bag_s
                WHERE sentence_id = $1
                ) AS b
        ORDER BY similarity DESC
        LIMIT 1000
);
$$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT sentence_id FROM bw.sentence_bag" "SELECT bw.build_sentence_similarity_graph (%s);" 8

CREATE INDEX sentence_similarity_tail ON bw.sentence_similarity (tail);

-- cluster 5
--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001,
	PRIMARY KEY (item_key)
);

CREATE SEQUENCE bw.cluster_five_seq MINVALUE 1 MAXVALUE 111082 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.cluster_five_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.sentence_similarity
		) AS a
	ORDER BY cluster_key
);--100 rows affected
CREATE INDEX itemclusters_clusterkey78 ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.sentence_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

CREATE OR REPLACE FUNCTION public.get_clustering_statistics()
	RETURNS TABLE(count INT4, maxsum FLOAT8, maxcount INT2, sum FLOAT8) AS $$
SELECT COUNT(*)::INT4, MAX(sum), MAX(count)::INT2, SUM(sum)
FROM    (
	SELECT cluster_key, COUNT(*), SUM(density)
	FROM public.itemclusters
	GROUP BY cluster_key
	ORDER BY sum DESC
	) AS foo
WHERE sum > 15
; $$ LANGUAGE 'SQL';

SELECT * FROM public.get_clustering_statistics()
ALTER TABLE public.cluster5 SET SCHEMA bw;

-- filter 

CREATE TABLE bw.cluster5_old AS (
	SELECT * FROM bw.cluster5
);


DELETE FROM bw.cluster5 AS a
USING	(
	SELECT cluster_key, SUM(density)
	FROM bw.cluster5 
	GROUP BY cluster_key
	ORDER BY sum ASC
	LIMIT 82
	) AS b
WHERE a.cluster_key = b.cluster_key;

	


-- cluster 4

--DROP TABLE bw.c4_sentence_bag;
CREATE TABLE bw.c4_sentence_bag (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c4_sentence_bag (cluster_key, word_id, word_freq) (
	SELECT b.cluster_key, a.word_id, SUM(word_freq)
	FROM bw.sentence_bag_s AS a, bw.cluster5 AS b
	WHERE a.sentence_id = b.item_key
	GROUP BY b.cluster_key, a.word_id
	ORDER BY b.cluster_key, sum DESC
);--11,669,545 rows affected 
CREATE INDEX c4_sentence_bag_cluster_key ON bw.c4_sentence_bag (cluster_key);
CREATE INDEX c4_sentence_bag_word_id ON bw.c4_sentence_bag(word_id);
ANALYSE bw.c4_sentence_bag;

CREATE TABLE bw.c4_sentence_bag_w (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c4_sentence_bag_w (cluster_key, word_id, word_freq) (
	SELECT cluster_key, word_id, word_freq
	FROM bw.c4_sentence_bag
	ORDER BY word_id
);
CREATE INDEX c4_sentence_bag_w_word_id ON bw.c4_sentence_bag_w (word_id);
ANALYSE bw.c4_sentence_bag_w;

--DROP TABLE bw.cluster5_similarity;
CREATE TABLE bw.cluster5_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_cluster5_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.cluster5_similarity (tail, head, similarity) (
        SELECT  1,
                cluster_key,
                dot_product
                /(
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c4_sentence_bag AS c
                WHERE c.cluster_key = a.cluster_key
                )
                *
                b.norm
                ) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.word_freq*b.word_freq)::FLOAT8 AS dot_product
                        FROM bw.c4_sentence_bag AS a, bw.c4_sentence_bag_w AS b
                        WHERE a.cluster_key = 1 AND a.word_id = b.word_id 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != 1
                ) AS a,
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c4_sentence_bag
                WHERE cluster_key = 1
                ) AS b
        ORDER BY similarity DESC
        LIMIT 300
);
$$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.cluster5" "SELECT bw.build_cluster5_similarity_graph (%s);" 8

CREATE INDEX cluster5_similarity_tail ON bw.cluster5_similarity (tail);


CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001,
	PRIMARY KEY (item_key)
);

CREATE SEQUENCE bw.cluster_four_seq MINVALUE 1 MAXVALUE 11100 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.cluster_four_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT cluster_key AS item_key
		FROM bw.cluster5
		) AS a
	ORDER BY cluster_key
);--100 rows affected
CREATE INDEX itemclusters_clusterkey79 ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.cluster5_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

CREATE OR REPLACE FUNCTION public.get_clustering_statistics()
	RETURNS TABLE(count INT4, maxsum FLOAT8, maxcount INT2, sum FLOAT8) AS $$
SELECT COUNT(*)::INT4, MAX(sum), MAX(count)::INT2, SUM(sum)
FROM    (
	SELECT cluster_key, COUNT(*), SUM(density)
	FROM public.itemclusters
	GROUP BY cluster_key
	ORDER BY sum DESC
	) AS foo
WHERE sum > 15
; $$ LANGUAGE 'SQL';

SELECT * FROM public.get_clustering_statistics()