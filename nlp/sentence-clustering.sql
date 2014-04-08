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
        SELECT  $1,
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
                        WHERE a.cluster_key = $1 AND a.word_id = b.word_id 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a,
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c4_sentence_bag
                WHERE cluster_key = $1
                ) AS b
        ORDER BY similarity DESC
        LIMIT 300
);
$$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.cluster5" "SELECT bw.build_cluster5_similarity_graph (%s);" 8

CREATE INDEX cluster5_similarity_tail ON bw.cluster5_similarity (tail);

--DROP TABLE public.itemclusters;
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
WHERE sum > 20
; $$ LANGUAGE 'SQL';

SELECT * FROM public.get_clustering_statistics()

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO cluster4;
CREATE INDEX cluster_4_pkey ON bw.cluster4 (item_key);

-- filter 

CREATE TABLE bw.cluster4_old AS (
	SELECT * FROM bw.cluster4
);


DELETE FROM bw.cluster4 AS a
USING	(
	SELECT cluster_key, SUM(density)
	FROM bw.cluster4 
	GROUP BY cluster_key
	ORDER BY sum ASC
	LIMIT 100
	) AS b
WHERE a.cluster_key = b.cluster_key;



-- cluster 3

--DROP TABLE bw.c3_sentence_bag;
CREATE TABLE bw.c3_sentence_bag (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c3_sentence_bag (cluster_key, word_id, word_freq) (
	SELECT c.cluster_key, a.word_id, SUM(word_freq) AS sum
	FROM bw.sentence_bag_s AS a, bw.cluster5 AS b, bw.cluster4 AS c
	WHERE a.sentence_id = b.item_key AND b.cluster_key = c.item_key
	GROUP BY c.cluster_key, a.word_id
	ORDER BY c.cluster_key, sum DESC
);--1414200 rows affected 
CREATE INDEX c3_sentence_bag_cluster_key ON bw.c3_sentence_bag (cluster_key);
ANALYSE bw.c3_sentence_bag;

--DROP TABLE bw.c3_sentence_bag_w;
CREATE TABLE bw.c3_sentence_bag_w (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c3_sentence_bag_w (cluster_key, word_id, word_freq) (
	SELECT cluster_key, word_id, word_freq
	FROM bw.c3_sentence_bag
	ORDER BY word_id
);
CREATE INDEX c3_sentence_bag_w_word_id ON bw.c3_sentence_bag_w (word_id);
ANALYSE bw.c3_sentence_bag_w;

--DROP TABLE bw.cluster4_similarity;
CREATE TABLE bw.cluster4_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_cluster4_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.cluster4_similarity (tail, head, similarity) (
        SELECT  $1,
                cluster_key,
                dot_product
                /(
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c3_sentence_bag AS c
                WHERE c.cluster_key = a.cluster_key
                )
                *
                b.norm
                ) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.word_freq*b.word_freq)::FLOAT8 AS dot_product
                        FROM bw.c3_sentence_bag AS a, bw.c3_sentence_bag_w AS b
                        WHERE a.cluster_key = $1 AND a.word_id = b.word_id 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a,
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c3_sentence_bag
                WHERE cluster_key = $1
                ) AS b
        ORDER BY similarity DESC
);
$$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.cluster4" "SELECT bw.build_cluster4_similarity_graph (%s);" 8

CREATE INDEX cluster4_similarity_tail ON bw.cluster4_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);

CREATE SEQUENCE bw.cluster_three_seq MINVALUE 1 MAXVALUE 1100 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.cluster_three_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT cluster_key AS item_key
		FROM bw.cluster4
		) AS a
	ORDER BY cluster_key
);--11000 rows affected
CREATE UNIQUE INDEX cluster3_pk2 ON public.itemclusters (item_key);
CREATE INDEX itemclusters_clusterkey7912 ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.cluster4_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key);

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-05 19:53:17.386124+00";0.0575186215208;1.08512141331412;0.186154970647345;2047.70467712079
"2014-04-05 20:41:23.724131+00";0.095237673007559;2.04877817801486;0.322305247240577;3545.35771964635
"2014-04-05 21:19:31.184944+00";0.0682979830009741;2.12268351541047;0.407805889347874;4485.86478282662
"2014-04-05 21:34:47.41843+00";0.02174466354466;3.06948827822965;0.452872176915734;4981.59394607308
"2014-04-06 01:29:53.909037+00";0.078817021080107;8.45696708327884;1.40940036553245;15503.4040208569
"2014-04-06 03:48:46.0324+00";0.0760203691015408;8.56153300191382;1.53773638290789;16915.1002119868
"2014-04-06 19:29:41.878177+00";0.0692059020912971;8.56153300191382;1.63648478829385;18001.3326712323
"2014-04-06 20:28:11.935348+00";0.076745781627483;8.56153300191382;1.63754370140416;18012.9807154458
"2014-04-06 22:11:56.838698+00";0.0673803861193394;8.56153300191382;1.63935340643479;18032.8874707827
"2014-04-07 05:23:13.970698+00";0.031605208505002;8.56153300191382;1.64887934104334;18137.6727514767
"2014-04-07 14:35:49.926356+00";0.0198147119797578;8.56153300191382;1.65392108939895;18193.1319833885
"2014-04-08 02:23:32.412861+00";0.0198147119797578;8.56153300191382;1.65386677913982;18192.534570538
*/


SELECT COUNT(*)
FROM	(
	SELECT DISTINCT tail 
	FROM bw.cluster4_similarity AS a
	) AS a, public.itemclusters AS b
WHERE a.tail = b.item_key;


ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO cluster3;

-- filter 

CREATE TABLE bw.cluster3_old AS (
	SELECT * FROM bw.cluster3
);

DELETE FROM bw.cluster3 AS a
USING	(
	SELECT cluster_key, SUM(density)
	FROM bw.cluster3
	GROUP BY cluster_key
	ORDER BY sum ASC
	LIMIT 50
	) AS b
WHERE a.cluster_key = b.cluster_key;



-- cluster 2

--DROP TABLE bw.c2_sentence_bag;
CREATE TABLE bw.c2_sentence_bag (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c2_sentence_bag (cluster_key, word_id, word_freq) (
	SELECT d.cluster_key, a.word_id, SUM(word_freq) AS sum
	FROM bw.sentence_bag_s AS a, bw.cluster5 AS b, bw.cluster4 AS c, bw.cluster3 AS d
	WHERE a.sentence_id = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key
	GROUP BY d.cluster_key, a.word_id
	ORDER BY d.cluster_key, sum DESC
);--5384532 rows affected 
CREATE INDEX c2_sentence_bag_cluster_key ON bw.c2_sentence_bag (cluster_key);
ANALYSE bw.c2_sentence_bag;

--DROP TABLE bw.c2_sentence_bag_w;
CREATE TABLE bw.c2_sentence_bag_w (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c2_sentence_bag_w (cluster_key, word_id, word_freq) (
	SELECT cluster_key, word_id, word_freq
	FROM bw.c2_sentence_bag
	ORDER BY word_id
);
CREATE INDEX c2_sentence_bag_w_word_id ON bw.c2_sentence_bag_w (word_id);
ANALYSE bw.c2_sentence_bag_w;

--DROP TABLE bw.cluster3_similarity;
CREATE TABLE bw.cluster3_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_cluster3_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.cluster3_similarity (tail, head, similarity) (
        SELECT  $1,
                cluster_key,
                dot_product
                /(
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c2_sentence_bag AS c
                WHERE c.cluster_key = a.cluster_key
                )
                *
                b.norm
                ) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.word_freq*b.word_freq)::FLOAT8 AS dot_product
                        FROM bw.c2_sentence_bag AS a, bw.c2_sentence_bag_w AS b
                        WHERE a.cluster_key = $1 AND a.word_id = b.word_id 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a,
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c2_sentence_bag
                WHERE cluster_key = $1
                ) AS b
        ORDER BY similarity DESC
);
$$ LANGUAGE 'SQL';
--here
python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.cluster3" "SELECT bw.build_cluster3_similarity_graph (%s);" 8

CREATE INDEX cluster3_similarity_tail ON bw.cluster3_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);

CREATE SEQUENCE bw.cluster_two_seq MINVALUE 1 MAXVALUE 105 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.cluster_two_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT cluster_key AS item_key
		FROM bw.cluster3
		) AS a
	ORDER BY cluster_key
);--1050 rows affected
CREATE UNIQUE INDEX cluster2_pk2 ON public.itemclusters (item_key);
CREATE INDEX itemclusters_clusterkey79125 ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.cluster4_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key);

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-08 14:42:06.172286+00";1e-16;0.515308481183997;0.169984076959764;178.483280807752
"2014-04-08 14:51:09.143397+00";0.172231869309608;3.25359933170708;1.32194354197734;1388.0407190762
"2014-04-08 14:55:03.232104+00";0.153876193768813;4.15125419165124;1.36186440899743;1429.9576294473
"2014-04-08 14:55:26.946119+00";0.248464577490603;4.15125419165124;1.36606079024211;1434.36382975422
"2014-04-08 15:00:05.569471+00";0.234357501520038;4.24137724181745;1.38585294491411;1455.14559215981
*/


SELECT COUNT(*)
FROM	(
	SELECT DISTINCT tail 
	FROM bw.cluster3_similarity AS a
	) AS a, public.itemclusters AS b
WHERE a.tail = b.item_key;


ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO cluster2;

-- filter 

CREATE TABLE bw.cluster2_old AS (
	SELECT * FROM bw.cluster2
);

DELETE FROM bw.cluster2 AS a
USING	(
	SELECT cluster_key, SUM(density)
	FROM bw.cluster2
	GROUP BY cluster_key
	ORDER BY sum ASC
	LIMIT 5
	) AS b
WHERE a.cluster_key = b.cluster_key;


-- cluster 1

--DROP TABLE bw.c1_sentence_bag;
CREATE TABLE bw.c1_sentence_bag (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c1_sentence_bag (cluster_key, word_id, word_freq) (
	SELECT e.cluster_key, a.word_id, SUM(word_freq) AS sum
	FROM bw.sentence_bag_s AS a, bw.cluster5 AS b, bw.cluster4 AS c, bw.cluster3 AS d, bw.cluster2 AS e
	WHERE a.sentence_id = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key AND d.cluster_key = e.item_key
	GROUP BY e.cluster_key, a.word_id
	ORDER BY e.cluster_key, sum DESC
);--2306419 rows affected 
CREATE INDEX c1_sentence_bag_cluster_key ON bw.c1_sentence_bag (cluster_key);
ANALYSE bw.c1_sentence_bag;

--DROP TABLE bw.c1_sentence_bag_w;
CREATE TABLE bw.c1_sentence_bag_w (
	cluster_key	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.c1_sentence_bag_w (cluster_key, word_id, word_freq) (
	SELECT cluster_key, word_id, word_freq
	FROM bw.c1_sentence_bag
	ORDER BY word_id
);
CREATE INDEX c1_sentence_bag_w_word_id ON bw.c1_sentence_bag_w (word_id);
ANALYSE bw.c1_sentence_bag_w;

--DROP TABLE bw.cluster2_similarity;
CREATE TABLE bw.cluster2_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_cluster2_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.cluster2_similarity (tail, head, similarity) (
        SELECT  $1,
                cluster_key,
                dot_product
                /(
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c1_sentence_bag AS c
                WHERE c.cluster_key = a.cluster_key
                )
                *
                b.norm
                ) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.word_freq*b.word_freq)::FLOAT8 AS dot_product
                        FROM bw.c1_sentence_bag AS a, bw.c1_sentence_bag_w AS b
                        WHERE a.cluster_key = $1 AND a.word_id = b.word_id 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a,
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.c1_sentence_bag
                WHERE cluster_key = $1
                ) AS b
        ORDER BY similarity DESC
);
$$ LANGUAGE 'SQL';
--here
python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.cluster2" "SELECT bw.build_cluster2_similarity_graph (%s);" 8

CREATE INDEX cluster2_similarity_tail ON bw.cluster2_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);

CREATE SEQUENCE bw.cluster_one_seq MINVALUE 1 MAXVALUE 10 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.cluster_one_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT cluster_key AS item_key
		FROM bw.cluster2
		) AS a
	ORDER BY cluster_key
);--100 rows affected
CREATE UNIQUE INDEX cluster1_pk2 ON public.itemclusters (item_key);
CREATE INDEX itemclusters_clusterkey79124 ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.cluster2_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key);

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-08 15:36:10.517609+00";1.24604992445927;4.6021959303321;2.64421779968726;264.421779968726
"2014-04-08 15:46:29.357924+00";1.33387736374846;5.4220344634975;3.1926837073989;319.26837073989
"2014-04-08 15:49:12.783516+00";1.25202178306952;5.27238814532094;3.22014240294651;322.014240294651
*/


ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO cluster1;
