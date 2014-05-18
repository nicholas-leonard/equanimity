CREATE OR REPLACE FUNCTION bw.filter_word2_similarity_graph ( item_key INT4 ) RETURNS VOID AS $$
DELETE FROM bw.word_similarity
USING bw.word2_cluster5 AS a, bw.word2_cluster4 AS b, bw.word2_cluster4 AS c, bw.word2_cluster5 AS d
WHERE tail = $1 AND a.item_key = tail AND a.cluster_key = b.item_key 
AND b.cluster_key = c.cluster_key AND c.item_key = d.cluster_key AND d.item_key = head;
$$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT word_id FROM bw.word_count" "SELECT bw.filter_word2_similarity_graph (%s);" 8
SELECT COUNT(*) FROM bw.word_similarity--503367310 vs 522312872  (lost 25 million)

-- missing some words... will require frequency based softmax
SELECT COUNT(*) FROM (SELECT DISTINCT target_word FROM bw.word_context) As a--783665
SELECT COUNT(*) FROM (SELECT DISTINCT tail FROM bw.word_similarity) As a--783622
SELECT COUNT(*) FROM bw.word_count --793471

-- word 3 cluster 5
--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);

CREATE SEQUENCE bw.word3_cluster5_seq MINVALUE 1 MAXVALUE 78362 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word3_cluster5_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word_similarity
		) AS a
	ORDER BY cluster_key
);--783622 rows affected
CREATE UNIQUE INDEX word3cluster5_itemkey ON public.itemclusters (item_key);
CREATE INDEX word3cluster5_clusterkey ON public.itemclusters (cluster_key);

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
"2014-05-08 19:31:31.630807-04";1e-16;0.935605644990462;0.0011750345575534;920.782930059108
"2014-05-08 19:34:54.761184-04";1e-16;0.935605644990462;0.0085025204185579;6662.76205543118
"2014-05-09 11:52:04.033798-04";1e-16;3.2544108802011;0.564374742914277;442256.464791971
"2014-05-09 11:57:08.631292-04";1e-16;3.77094821287227;0.567624100243519;444802.732681027
"2014-05-09 12:08:33.211719-04";1e-16;3.98326837850898;0.574918370651597;450518.683446746
"2014-05-09 12:35:49.984489-04";1e-16;5.48174601889831;0.592151007947812;464022.55715008
"2014-05-09 13:50:38.401707-04";1e-16;5.92543243740915;0.642077668261911;503146.186558735
"2014-05-09 14:38:46.095505-04";1e-16;8.15008441520778;0.669585901633554;524702.243409889
"2014-05-09 15:36:02.096635-04";1e-16;8.43427450499916;0.69140643780839;541801.295608286
"2014-05-09 16:01:19.169159-04";1e-16;8.43427450499916;0.694963728327384;544588.866719361
"2014-05-09 18:21:38.348139-04";1e-16;8.47714838471228;0.721537950174181;565413.011591392
"2014-05-11 01:46:45.389813-04";1e-16;8.68099808910186;1.05171148355356;824144.256165205
"2014-05-12 18:18:17.786425-04";1e-16;9.16115316268726;1.13593228410017;890141.52833114
"2014-05-12 18:47:29.940242-04";1e-16;8.85990579399145;1.13670159114976;890744.374259956
"2014-05-12 19:59:08.073199-04";1e-16;8.85990579399145;1.13725657171265;891179.269238608
"2014-05-13 17:40:25.88482-04";1e-16;8.92021055400821;1.15063281713788;901661.189431217

"2014-04-25 20:26:55.418564-04";1e-16;9.23546905571801;1.35203867255162;1059494.00885561
*/

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word3_cluster5;



--cluster 4
--DROP TABLE bw.c4_word3_context;
CREATE TABLE bw.c4_word3_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c4_word3_context (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word3_cluster5 AS b
	WHERE a.target_word = b.item_key
	GROUP BY b.cluster_key, a.context_word
	ORDER BY b.cluster_key, sum DESC
);--69,100,394 rows affected
CREATE INDEX c4_word3_context_clusterkey ON bw.c4_word3_context (cluster_key);
ANALYSE bw.c4_word3_context;

--DROP TABLE bw.c4_word3_context_c;
CREATE TABLE bw.c4_word3_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c4_word3_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c4_word3_context
	ORDER BY context_word
);--69100394 rows affected
CREATE INDEX c4_word3_context_contextword ON bw.c4_word3_context_c (context_word);
ANALYSE bw.c4_word3_context_c;

--DROP TABLE bw.c4_idf_norm3;
CREATE TABLE bw.c4_idf_norm3 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c4_idf_norm3 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c4_word3_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word3_cluster5_similarity;
CREATE TABLE bw.word3_cluster5_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word3_cluster5_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word3_cluster5_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c4_word3_context AS a, bw.c4_word3_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c4_idf_norm3 AS b, bw.c4_idf_norm3 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 500
); $$ LANGUAGE 'SQL';
--here
python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word3_cluster5" "SELECT bw.build_word3_cluster5_similarity_graph (%s);" 8

CREATE INDEX word3_cluster5_similarity_tail ON bw.word3_cluster5_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word3_cluster4_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word_cluster4_seq MINVALUE 1 MAXVALUE 7836 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster4_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word3_cluster5_similarity
		) AS a
	ORDER BY cluster_key
);--78363 rows affected
CREATE INDEX word3_cluster4_clusterkey ON public.itemclusters (cluster_key);


CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word3_cluster5_similarity AS a, public.itemclusters AS b
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
"2014-05-14 17:49:52.193304-04";1e-16;1.46801730666244;0.0101954039520282;798.932244488837
"2014-05-16 13:28:25.710896-04";1e-16;9.35646880719684;2.77884766961072;217756.061086035


"2014-04-05 16:43:31.76777-04";1e-16;8.89088499019322;2.79872595114696;219319.36043568
*/


ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word3_cluster4;



--cluster 3
--DROP TABLE bw.c3_word3_context;
CREATE TABLE bw.c3_word3_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c3_word3_context (cluster_key, context_word, tfidf) (
	SELECT c.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word3_cluster5 AS b, bw.word3_cluster4 AS c
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key
	GROUP BY c.cluster_key, a.context_word
	ORDER BY c.cluster_key, sum DESC
);--45278801 rows affected
CREATE INDEX c3_word3_context_clusterkey ON bw.c3_word3_context (cluster_key);
ANALYSE bw.c3_word3_context;

--DROP TABLE bw.c3_word3_context_c;
CREATE TABLE bw.c3_word3_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c3_word3_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c3_word3_context
	ORDER BY context_word
);--45278801 rows affected
CREATE INDEX c3_word3_context_contextword ON bw.c3_word3_context_c (context_word);
ANALYSE bw.c3_word3_context_c;

--DROP TABLE bw.c3_idf_norm3;
CREATE TABLE bw.c3_idf_norm3 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c3_idf_norm3 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c3_word3_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word3_cluster4_similarity;
CREATE TABLE bw.word3_cluster4_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word3_cluster4_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word3_cluster4_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c3_word3_context AS a, bw.c3_word3_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c3_idf_norm3 AS b, bw.c3_idf_norm3 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 400
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word3_cluster4" "SELECT bw.build_word3_cluster4_similarity_graph (%s);" 8

CREATE INDEX word3_cluster4_similarity_tail ON bw.word3_cluster4_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word3_cluster3_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word3_cluster3_seq MINVALUE 1 MAXVALUE 783 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster3_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word3_cluster4_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word3_cluster3_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word3_cluster4_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-05-16 16:37:15.995097-04";1e-16;1.58286217198339;0.0860472818339346;674.266500450712
"2014-05-16 17:34:17.215468-04";1e-16;8.62024151771346;2.42939013388534;19036.7010891256
"2014-05-16 17:51:05.293558-04";1e-16;8.69441843518634;2.5276134648764;19806.3791107715

"2014-05-02 23:30:11.265608-04";1e-16;9.08722091725535;2.52448191660131;19781.8402984879


"2014-04-07 22:09:01.090834-04";1e-16;9.83727072442732;2.42089858020349;18970.1612744745
*/



ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word3_cluster3;

--here

--cluster 2
--DROP TABLE bw.c2_word3_context;
CREATE TABLE bw.c2_word3_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c2_word3_context (cluster_key, context_word, tfidf) (
	SELECT d.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word3_cluster5 AS b, bw.word3_cluster4 AS c, bw.word3_cluster3 AS d
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key
	GROUP BY d.cluster_key, a.context_word
	ORDER BY d.cluster_key, sum DESC
);--22414450 rows affected
CREATE INDEX c2_word3_context_clusterkey ON bw.c2_word3_context (cluster_key);
ANALYSE bw.c2_word3_context;

--DROP TABLE bw.c2_word3_context_c;
CREATE TABLE bw.c2_word3_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c2_word3_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c2_word3_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c2_word3_context_contextword ON bw.c2_word3_context_c (context_word);
ANALYSE bw.c2_word3_context_c;

--DROP TABLE bw.c2_idf_norm3;
CREATE TABLE bw.c2_idf_norm3 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c2_idf_norm3 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c2_word3_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word3_cluster3_similarity;
CREATE TABLE bw.word3_cluster3_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word3_cluster3_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word3_cluster3_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c2_word3_context AS a, bw.c2_word3_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c2_idf_norm3 AS b, bw.c2_idf_norm3 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 300
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word3_cluster3" "SELECT bw.build_word3_cluster3_similarity_graph (%s);" 8

CREATE INDEX word3_cluster3_similarity_tail ON bw.word3_cluster3_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word3_cluster2_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word3_cluster2_seq MINVALUE 1 MAXVALUE 78 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster2_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word3_cluster3_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word3_cluster2_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word3_cluster3_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-05-17 21:49:55.840871-04";1e-16;2.63798407606359;0.696545585572221;545.395193503049
"2014-05-17 22:47:10.93614-04";0.0393659770583017;8.31576930024687;2.47277473961206;1936.18262111624
"2014-05-18 10:21:11.101897-04";1e-16;8.27088818313764;2.50068998812254;1958.04026069995

"2014-05-03 00:17:17.152074-04";1e-16;7.43448636635493;2.34153083315739;1833.41864236224

"2014-04-08 10:53:14.984615-04";1e-16;8.10712832858916;2.38806904288253;1869.85806057702
*/

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word3_cluster2;


--cluster 1
--DROP TABLE bw.c1_word3_context;
CREATE TABLE bw.c1_word3_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c1_word3_context (cluster_key, context_word, tfidf) (
	SELECT e.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word3_cluster5 AS b, bw.word3_cluster4 AS c, bw.word3_cluster3 AS d, bw.word3_cluster2 AS e
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key AND d.cluster_key = e.item_key
	GROUP BY e.cluster_key, a.context_word
	ORDER BY e.cluster_key, sum DESC
);--8948825 rows affected
CREATE INDEX c1_word3_context_clusterkey ON bw.c1_word3_context (cluster_key);
ANALYSE bw.c1_word3_context;

--DROP TABLE bw.c1_word3_context_c;
CREATE TABLE bw.c1_word3_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c1_word3_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c1_word3_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c1_word3_context_contextword ON bw.c1_word3_context_c (context_word);
ANALYSE bw.c1_word3_context_c;

--DROP TABLE bw.c1_idf_norm3;
CREATE TABLE bw.c1_idf_norm3 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c1_idf_norm3 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c1_word3_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word3_cluster2_similarity;
CREATE TABLE bw.word3_cluster2_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);

--here
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word3_cluster2_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word3_cluster2_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c1_word3_context AS a, bw.c1_word3_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c1_idf_norm3 AS b, bw.c1_idf_norm3 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word3_cluster2" "SELECT bw.build_word3_cluster2_similarity_graph (%s);" 8

CREATE INDEX word3_cluster2_similarity_tail ON bw.word3_cluster2_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word3_cluster1_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word3_cluster1_seq MINVALUE 1 MAXVALUE 8 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster1_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word3_cluster2_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word3_cluster1_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word3_cluster2_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-05-03 19:00:49.670045-04";0.679805119013395;5.08621007219643;2.8778059505398;224.468864142105
"2014-05-04 15:08:41.399982-04";0.398090259563034;9.31274572480048;4.02156683126238;313.682212838466
            
"2014-04-08 11:50:08.410578-04";0.819812222942905;9.38691881957718;5.50862530709135;429.672773953125
"2014-04-08 15:00:44.734685-04";0.521600890890975;15.0471630571185;7.26388062798665;566.582688982959
*/

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word3_cluster1;