CREATE OR REPLACE FUNCTION bw.filter_word_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
DELETE FROM bw.word_similarity
USING bw.word_cluster5 AS a, bw.word_cluster4 AS b, bw.word_cluster4 AS c, bw.word_cluster5 AS d
WHERE tail = $1 AND a.item_key = tail AND a.cluster_key = b.item_key 
AND b.cluster_key = c.cluster_key AND c.item_key = d.cluster_key AND d.item_key = head;
$$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT word_id FROM bw.word_count" "SELECT bw.filter_word_similarity_graph (%s);" 8

-- missing some words... will require frequency based softmax
SELECT COUNT(*) FROM (SELECT DISTINCT target_word FROM bw.word_context) As a--783665
SELECT COUNT(*) FROM (SELECT DISTINCT tail FROM bw.word_similarity) As a--783627
SELECT COUNT(*) FROM bw.word_count --793471

-- word 2 cluster 5
--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);

CREATE SEQUENCE bw.word2_cluster5_seq MINVALUE 1 MAXVALUE 78363 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word2_cluster5_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word_similarity
		) AS a
	ORDER BY cluster_key
);--783627 rows affected
CREATE UNIQUE INDEX word2cluster5_itemkey ON public.itemclusters (item_key);
CREATE INDEX word2cluster5_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-09 15:41:39.512171-04";1e-16;0.998634399160141;0.00127002806040624;995.228278891958
"2014-04-09 16:00:07.146723-04";1e-16;0.998634399160141;0.0109082441366237;8547.99462805003
"2014-04-09 16:54:16.038137-04";1e-16;1.18191262267853;0.0321326342444013;25179.9997750375
"2014-04-09 17:08:08.475929-04";1e-16;1.18191262267853;0.0345313085185869;27059.6657004947
"2014-04-10 01:41:18.213963-04";1e-16;2.24469519624583;0.260408305435386;204062.979163415
"2014-04-10 12:21:26.626992-04";1e-16;6.86948064390496;0.678296509956417;531531.459207617
"2014-04-10 15:43:24.634704-04";1e-16;8.47984544231353;0.735453834469036;576321.481943467
*/

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word_cluster5;



--cluster 4
--DROP TABLE bw.c4_word_context;
CREATE TABLE bw.c4_word_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c4_word_context (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word_cluster5 AS b
	WHERE a.target_word = b.item_key
	GROUP BY b.cluster_key, a.context_word
	ORDER BY b.cluster_key, sum DESC
);--67,817,855 rows affected
CREATE INDEX c4_word_context_clusterkey ON bw.c4_word_context (cluster_key);
ANALYSE bw.c4_word_context;

--DROP TABLE bw.c4_word_context_c;
CREATE TABLE bw.c4_word_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c4_word_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c4_word_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c4_word_context_contextword ON bw.c4_word_context_c (context_word);
ANALYSE bw.c4_sentence_bag;

--DROP TABLE bw.c4_idf_norm;
CREATE TABLE bw.c4_idf_norm (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c4_idf_norm (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c4_word_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word_cluster5_similarity;
CREATE TABLE bw.word_cluster5_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word_cluster5_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word_cluster5_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c4_word_context AS a, bw.c4_word_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c4_idf_norm AS b, bw.c4_idf_norm AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 500
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word_cluster5" "SELECT bw.build_word_cluster5_similarity_graph (%s);" 8

CREATE INDEX word_cluster5_similarity_tail ON bw.word_cluster5_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word_cluster4_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word_cluster4_seq MINVALUE 1 MAXVALUE 7836 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster4_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word_cluster5_similarity
		) AS a
	ORDER BY cluster_key
);--78364 rows affected
CREATE INDEX word_cluster4_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word_cluster5_similarity AS a, public.itemclusters AS b
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

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-03 22:05:17.554822-04";1e-16;1.34388094320337;0.00966920609890062;757.717666734248
"2014-04-03 22:45:20.096571-04";1e-16;1.64984054794932;0.299713062079679;23486.714396812
"2014-04-03 23:50:17.88992-04";1e-16;2.50716473855602;0.65221781922496;51,110.3971857447
"2014-04-04 15:22:24.9507-04";1e-16;8.89088499019322;2.77875208498729;217,754.128387944
"2014-04-04 15:58:57.70527-04";1e-16;8.89088499019322;2.78419864043016;218180.942258669
"2014-04-04 16:25:02.766524-04";1e-16;8.89088499019322;2.78492217859654;218237.641603539
"2014-04-05 13:03:15.104483-04";1e-16;8.89830242260854;2.79580842831344;219090.731676354
"2014-04-05 16:41:43.65326-04";1e-16;8.89088499019322;2.79779107500886;219246.099801995
"2014-04-05 16:43:31.76777-04";1e-16;8.89088499019322;2.79872595114696;219319.36043568
*/


ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word_cluster4;



--cluster 3
--DROP TABLE bw.c3_word_context;
CREATE TABLE bw.c3_word_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c3_word_context (cluster_key, context_word, tfidf) (
	SELECT c.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word_cluster5 AS b, bw.word_cluster4 AS c
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key
	GROUP BY c.cluster_key, a.context_word
	ORDER BY c.cluster_key, sum DESC
);--44,867,261 rows affected
CREATE INDEX c3_word_context_clusterkey ON bw.c3_word_context (cluster_key);
ANALYSE bw.c3_word_context;

--DROP TABLE bw.c3_word_context_c;
CREATE TABLE bw.c3_word_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c3_word_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c3_word_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c3_word_context_contextword ON bw.c3_word_context_c (context_word);
ANALYSE bw.c3_word_context_c;

--DROP TABLE bw.c3_idf_norm;
CREATE TABLE bw.c3_idf_norm (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c3_idf_norm (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c3_word_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word_cluster4_similarity;
CREATE TABLE bw.word_cluster4_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word_cluster4_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word_cluster4_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c3_word_context AS a, bw.c3_word_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c3_idf_norm AS b, bw.c3_idf_norm AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 400
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word_cluster4" "SELECT bw.build_word_cluster4_similarity_graph (%s);" 8

CREATE INDEX word_cluster4_similarity_tail ON bw.word_cluster4_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word_cluster3_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word_cluster3_seq MINVALUE 1 MAXVALUE 783 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster3_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word_cluster4_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word_cluster3_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word_cluster4_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-05 23:48:08.992031-04";1e-16;8.76426693329261;2.39399654385733;18759.356917666
"2014-04-06 15:30:11.056913-04";1e-16;8.76426693329261;2.42101744293811;18971.092682863
"2014-04-06 16:27:38.891373-04";1e-16;8.76426693329261;2.43825247890122;19106.1464246699
"2014-04-06 18:11:48.465287-04";1e-16;8.76426693329261;2.42022018338484;18964.8453570036
"2014-04-07 01:22:16.78404-04";1e-16;8.76426693329261;2.42241796030375;18982.0671369402
"2014-04-07 10:36:06.99847-04";1e-16;9.83727072442732;2.41417024526963;18917.4380419328
"2014-04-07 22:09:01.090834-04";1e-16;9.83727072442732;2.42089858020349;18970.1612744745
*/



ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word_cluster3;



--cluster 2
--DROP TABLE bw.c2_word_context;
CREATE TABLE bw.c2_word_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c2_word_context (cluster_key, context_word, tfidf) (
	SELECT d.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word_cluster5 AS b, bw.word_cluster4 AS c, bw.word_cluster3 AS d
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key
	GROUP BY d.cluster_key, a.context_word
	ORDER BY d.cluster_key, sum DESC
);--22414450 rows affected
CREATE INDEX c2_word_context_clusterkey ON bw.c2_word_context (cluster_key);
ANALYSE bw.c2_word_context;

--DROP TABLE bw.c2_word_context_c;
CREATE TABLE bw.c2_word_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c2_word_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c2_word_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c2_word_context_contextword ON bw.c2_word_context_c (context_word);
ANALYSE bw.c2_word_context_c;

--DROP TABLE bw.c2_idf_norm;
CREATE TABLE bw.c2_idf_norm (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c2_idf_norm (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c2_word_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word_cluster3_similarity;
CREATE TABLE bw.word_cluster3_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word_cluster3_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word_cluster3_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c2_word_context AS a, bw.c2_word_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c2_idf_norm AS b, bw.c2_idf_norm AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 300
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word_cluster3" "SELECT bw.build_word_cluster3_similarity_graph (%s);" 8

CREATE INDEX word_cluster3_similarity_tail ON bw.word_cluster3_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word_cluster2_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word_cluster2_seq MINVALUE 1 MAXVALUE 78 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster2_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word_cluster3_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word_cluster2_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word_cluster3_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-08 10:36:42.346651-04";1e-16;1.366318096596;0.0781080222670765;61.1585814351209
"2014-04-08 10:51:21.119347-04";0.0511003623204846;7.67891425693711;2.32420540537846;1819.85283241134
"2014-04-08 10:53:14.984615-04";1e-16;8.10712832858916;2.38806904288253;1869.85806057702
*/

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word_cluster2;


--cluster 1
--DROP TABLE bw.c1_word_context;
CREATE TABLE bw.c1_word_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c1_word_context (cluster_key, context_word, tfidf) (
	SELECT e.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word_cluster5 AS b, bw.word_cluster4 AS c, bw.word_cluster3 AS d, bw.word_cluster2 AS e
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key AND d.cluster_key = e.item_key
	GROUP BY e.cluster_key, a.context_word
	ORDER BY e.cluster_key, sum DESC
);--8948825 rows affected
CREATE INDEX c1_word_context_clusterkey ON bw.c1_word_context (cluster_key);
ANALYSE bw.c1_word_context;

--DROP TABLE bw.c1_word_context_c;
CREATE TABLE bw.c1_word_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c1_word_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c1_word_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c1_word_context_contextword ON bw.c1_word_context_c (context_word);
ANALYSE bw.c1_word_context_c;

--DROP TABLE bw.c1_idf_norm;
CREATE TABLE bw.c1_idf_norm (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c1_idf_norm (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c1_word_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word_cluster2_similarity;
CREATE TABLE bw.word_cluster2_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word_cluster2_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word_cluster2_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c1_word_context AS a, bw.c1_word_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c1_idf_norm AS b, bw.c1_idf_norm AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word_cluster2" "SELECT bw.build_word_cluster2_similarity_graph (%s);" 8

CREATE INDEX word_cluster2_similarity_tail ON bw.word_cluster2_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word_cluster1_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word_cluster1_seq MINVALUE 1 MAXVALUE 8 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster1_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word_cluster2_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word_cluster1_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word_cluster2_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-04-08 11:50:08.410578-04";0.819812222942905;9.38691881957718;5.50862530709135;429.672773953125
"2014-04-08 15:00:44.734685-04";0.521600890890975;15.0471630571185;7.26388062798665;566.582688982959
*/

