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
"2014-04-10 20:22:16.102876-04";1e-16;8.61709107340828;0.910020993421114;713117.021011607
"2014-04-11 01:00:07.899902-04";1e-16;8.61709857023276;1.02530778551985;803458.864043564
"2014-04-11 13:15:35.899163-04";1e-16;8.54812299196085;1.15449015331455;904689.65537142
"2014-04-11 13:56:11.749511-04";1e-16;8.60761834804777;1.15918763336051;908370.727567394
"2014-04-11 14:47:31.867328-04";1e-16;9;1.16625860360042;913911.730763589
"2014-04-12 00:42:24.337217-04";1e-16;9;1.2139573425923;951289.750503575
"2014-04-12 01:49:58.204233-04";1e-16;9;1.21782355859076;954319.421747798
"2014-04-12 11:12:13.44369-04";1e-16;9;1.25281497052978;981739.636911343
"2014-04-12 16:28:04.805725-04";1e-16;8.88738419329539;1.26354770645116;990150.098563206
"2014-04-14 09:47:59.613255-04";1e-16;8.88738419329539;1.31319310326145;1029053.57192946
"2014-04-14 23:52:04.243865-04";1e-16;8.93086242723555;1.32104564431795;1035207.03511994
"2014-04-16 01:49:50.452325-04";1e-16;8.97428267140165;1.3319542353358;1043755.30157349
"2014-04-25 20:26:55.418564-04";1e-16;9.23546905571801;1.35203867255162;1059494.00885561
*/

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word2_cluster5;



--cluster 4
--DROP TABLE bw.c4_word2_context;
CREATE TABLE bw.c4_word2_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c4_word2_context (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word2_cluster5 AS b
	WHERE a.target_word = b.item_key
	GROUP BY b.cluster_key, a.context_word
	ORDER BY b.cluster_key, sum DESC
);--68,610,174 rows affected
CREATE INDEX c4_word2_context_clusterkey ON bw.c4_word2_context (cluster_key);
ANALYSE bw.c4_word2_context;

--DROP TABLE bw.c4_word2_context_c;
CREATE TABLE bw.c4_word2_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c4_word2_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c4_word2_context
	ORDER BY context_word
);--68610174 rows affected
CREATE INDEX c4_word2_context_contextword ON bw.c4_word2_context_c (context_word);
ANALYSE bw.c4_word2_context_c;

--DROP TABLE bw.c4_idf_norm2;
CREATE TABLE bw.c4_idf_norm2 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c4_idf_norm2 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c4_word2_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word2_cluster5_similarity;
CREATE TABLE bw.word2_cluster5_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word2_cluster5_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word2_cluster5_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c4_word2_context AS a, bw.c4_word2_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c4_idf_norm2 AS b, bw.c4_idf_norm2 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 500
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word2_cluster5" "SELECT bw.build_word2_cluster5_similarity_graph (%s);" 8

CREATE INDEX word2_cluster5_similarity_tail ON bw.word2_cluster5_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word2_cluster4_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word_cluster4_seq MINVALUE 1 MAXVALUE 7836 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster4_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word2_cluster5_similarity
		) AS a
	ORDER BY cluster_key
);--78363 rows affected
CREATE INDEX word2_cluster4_clusterkey ON public.itemclusters (cluster_key);


CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word2_cluster5_similarity AS a, public.itemclusters AS b
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
"2014-04-29 12:46:41.629531-04";1e-16;0.865878040939167;0.00979954449220231;767.92170504245
"2014-04-29 13:31:44.677453-04";1e-16;2.89790222413997;0.608025426090394;47646.6964647215
"2014-04-29 14:52:54.879974-04";1e-16;7.80527206996967;1.90421270543633;149219.820236107
"2014-04-29 15:04:53.867237-04";1e-16;8.12839816614851;2.01705101010628;158062.168304958
"2014-04-29 15:18:18.51372-04";1e-16;8.42973613093346;2.11426603985267;165680.229680975
"2014-04-29 16:57:58.816247-04";1e-16;8.78038314968583;2.46633124302471;193269.115197145
"2014-04-29 17:57:02.108965-04";1e-16;8.77862754409603;2.53857122423766;198930.056844935
"2014-04-29 20:41:37.443704-04";1e-16;8.86509586773414;2.62659069684557;205827.52677691
"2014-04-29 21:40:54.156734-04";1e-16;8.86509586773414;2.6432630529976;207134.022622051
"2014-04-29 21:58:53.300785-04";1e-16;8.86509586773414;2.6480839972877;207511.806279456
"2014-04-30 11:49:37.061039-04";1e-16;8.86509586773414;2.69278263494488;211014.525622185
"2014-05-01 13:27:35.820447-04";1e-16;8.88155079480399;2.70437434637516;211922.886904997

"2014-04-05 16:43:31.76777-04";1e-16;8.89088499019322;2.79872595114696;219319.36043568
*/


ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word2_cluster4;



--cluster 3
--DROP TABLE bw.c3_word2_context;
CREATE TABLE bw.c3_word2_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c3_word2_context (cluster_key, context_word, tfidf) (
	SELECT c.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word2_cluster5 AS b, bw.word2_cluster4 AS c
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key
	GROUP BY c.cluster_key, a.context_word
	ORDER BY c.cluster_key, sum DESC
);--45278801 rows affected
CREATE INDEX c3_word2_context_clusterkey ON bw.c3_word2_context (cluster_key);
ANALYSE bw.c3_word2_context;

--DROP TABLE bw.c3_word2_context_c;
CREATE TABLE bw.c3_word2_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c3_word2_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c3_word2_context
	ORDER BY context_word
);--45278801 rows affected
CREATE INDEX c3_word2_context_contextword ON bw.c3_word2_context_c (context_word);
ANALYSE bw.c3_word2_context_c;

--DROP TABLE bw.c3_idf_norm2;
CREATE TABLE bw.c3_idf_norm2 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c3_idf_norm2 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c3_word2_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word2_cluster4_similarity;
CREATE TABLE bw.word2_cluster4_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word2_cluster4_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word2_cluster4_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c3_word2_context AS a, bw.c3_word2_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c3_idf_norm2 AS b, bw.c3_idf_norm2 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 400
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word2_cluster4" "SELECT bw.build_word2_cluster4_similarity_graph (%s);" 8

CREATE INDEX word2_cluster4_similarity_tail ON bw.word2_cluster4_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word2_cluster3_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word2_cluster3_seq MINVALUE 1 MAXVALUE 783 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster3_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word2_cluster4_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word2_cluster3_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word2_cluster4_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-05-01 19:56:56.27931-04";1e-16;1.70896755913616;0.0832581384767601;652.410773103892
"2014-05-01 20:06:33.703882-04";1e-16;3.73408770922918;0.978401415775774;7666.75349401897
"2014-05-01 20:11:37.18543-04";1e-16;5.7206912515936;1.52517191228217;11951.2471046431
"2014-05-01 20:30:55.956749-04";1e-16;8.45338694130515;2.25103756996691;17639.1303982607
"2014-05-01 21:42:30.198112-04";1e-16;9.07496052372545;2.50643185105546;19640.3999848706
"2014-05-01 22:24:44.7539-04";1e-16;9.08722091725535;2.50875503202752;19658.6044309676
"2014-05-01 22:33:51.498927-04";1e-16;9.08722091725535;2.52113688739596;19755.6286496348
"2014-05-01 23:09:22.818196-04";1e-16;9.08722091725535;2.52894224027443;19816.7913947904
"2014-05-01 23:43:07.457632-04";1e-16;9.08722091725535;2.5440812230625;19935.4204639177
"2014-05-02 23:30:11.265608-04";1e-16;9.08722091725535;2.52448191660131;19781.8402984879


"2014-04-07 22:09:01.090834-04";1e-16;9.83727072442732;2.42089858020349;18970.1612744745
*/



ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word2_cluster3;



--cluster 2
--DROP TABLE bw.c2_word2_context;
CREATE TABLE bw.c2_word2_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c2_word2_context (cluster_key, context_word, tfidf) (
	SELECT d.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word2_cluster5 AS b, bw.word2_cluster4 AS c, bw.word2_cluster3 AS d
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key
	GROUP BY d.cluster_key, a.context_word
	ORDER BY d.cluster_key, sum DESC
);--22414450 rows affected
CREATE INDEX c2_word2_context_clusterkey ON bw.c2_word2_context (cluster_key);
ANALYSE bw.c2_word2_context;

--DROP TABLE bw.c2_word2_context_c;
CREATE TABLE bw.c2_word2_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c2_word2_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c2_word2_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c2_word2_context_contextword ON bw.c2_word2_context_c (context_word);
ANALYSE bw.c2_word2_context_c;

--DROP TABLE bw.c2_idf_norm2;
CREATE TABLE bw.c2_idf_norm2 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c2_idf_norm2 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c2_word2_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word2_cluster3_similarity;
CREATE TABLE bw.word2_cluster3_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word2_cluster3_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word2_cluster3_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c2_word2_context AS a, bw.c2_word2_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c2_idf_norm2 AS b, bw.c2_idf_norm2 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
        LIMIT 300
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word2_cluster3" "SELECT bw.build_word2_cluster3_similarity_graph (%s);" 8

CREATE INDEX word2_cluster3_similarity_tail ON bw.word2_cluster3_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word2_cluster2_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word2_cluster2_seq MINVALUE 1 MAXVALUE 78 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster2_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word2_cluster3_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word2_cluster2_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word2_cluster3_similarity AS a, public.itemclusters AS b
WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
$$ LANGUAGE 'SQL';

UPDATE public.itemclusters SET density = public.measure_density(item_key, cluster_key)

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-05-03 00:10:36.171277-04";1e-16;2.81614212583415;0.69873944946332;547.112988929779
"2014-05-03 00:17:17.152074-04";1e-16;7.43448636635493;2.34153083315739;1833.41864236224

"2014-04-08 10:53:14.984615-04";1e-16;8.10712832858916;2.38806904288253;1869.85806057702
*/

ALTER TABLE public.itemclusters SET SCHEMA bw;
ALTER TABLE bw.itemclusters RENAME TO word2_cluster2;


--cluster 1
--DROP TABLE bw.c1_word2_context;
CREATE TABLE bw.c1_word2_context (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c1_word2_context (cluster_key, context_word, tfidf) (
	SELECT e.cluster_key, context_word, SUM(tfidf)
	FROM bw.word_context AS a, bw.word2_cluster5 AS b, bw.word2_cluster4 AS c, bw.word2_cluster3 AS d, bw.word2_cluster2 AS e
	WHERE a.target_word = b.item_key AND b.cluster_key = c.item_key AND c.cluster_key = d.item_key AND d.cluster_key = e.item_key
	GROUP BY e.cluster_key, a.context_word
	ORDER BY e.cluster_key, sum DESC
);--8948825 rows affected
CREATE INDEX c1_word2_context_clusterkey ON bw.c1_word2_context (cluster_key);
ANALYSE bw.c1_word2_context;

--DROP TABLE bw.c1_word2_context_c;
CREATE TABLE bw.c1_word2_context_c (
	cluster_key    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.c1_word2_context_c (cluster_key, context_word, tfidf) (
	SELECT cluster_key, context_word, tfidf
	FROM bw.c1_word2_context
	ORDER BY context_word
);--67,817,855 rows affected
CREATE INDEX c1_word2_context_contextword ON bw.c1_word2_context_c (context_word);
ANALYSE bw.c1_word2_context_c;

--DROP TABLE bw.c1_idf_norm2;
CREATE TABLE bw.c1_idf_norm2 (
	cluster_key		INT4,
	cluster_idf_norm 	FLOAT8,
	PRIMARY KEY (cluster_key)
);
INSERT INTO bw.c1_idf_norm2 (cluster_key, cluster_idf_norm) (
	SELECT cluster_key, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.c1_word2_context AS c
	GROUP BY cluster_key
);

--DROP TABLE bw.word2_cluster2_similarity;
CREATE TABLE bw.word2_cluster2_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);

--here
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word2_cluster2_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word2_cluster2_similarity (tail, head, similarity) (
        SELECT  $1,
                a.cluster_key,
                dot_product/(c.cluster_idf_norm * b.cluster_idf_norm) AS similarity                
        FROM    (
                SELECT cluster_key, dot_product
                FROM    (
                        SELECT b.cluster_key, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.c1_word2_context AS a, bw.c1_word2_context_c AS b
                        WHERE a.cluster_key = $1 AND a.context_word = b.context_word 
                        GROUP BY b.cluster_key
                        ) AS a
                WHERE cluster_key != $1
                ) AS a, bw.c1_idf_norm2 AS b, bw.c1_idf_norm2 AS c
        WHERE b.cluster_key = $1 AND c.cluster_key = a.cluster_key
        ORDER BY similarity DESC
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT DISTINCT cluster_key FROM bw.word2_cluster2" "SELECT bw.build_word2_cluster2_similarity_graph (%s);" 8

CREATE INDEX word2_cluster2_similarity_tail ON bw.word2_cluster2_similarity (tail);

--DROP TABLE public.itemclusters;
CREATE TABLE public.itemclusters (
	item_key	INT4,
	cluster_key	INT4,
	density		FLOAT8 DEFAULT 0.00000001
);
CREATE INDEX word2_cluster1_itemkey ON public.itemclusters (item_key);

CREATE SEQUENCE bw.word2_cluster1_seq MINVALUE 1 MAXVALUE 8 CYCLE;
INSERT INTO public.itemclusters (item_key, cluster_key) (
	SELECT item_key, nextval('bw.word_cluster1_seq') AS cluster_key
	FROM	(
		SELECT DISTINCT tail AS item_key
		FROM bw.word2_cluster2_similarity
		) AS a
	ORDER BY cluster_key
);--7836 rows affected
CREATE INDEX word2_cluster1_clusterkey ON public.itemclusters (cluster_key);

CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
    RETURNS FLOAT8 AS $$
SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
FROM bw.word2_cluster2_similarity AS a, public.itemclusters AS b
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
ALTER TABLE bw.itemclusters RENAME TO word2_cluster1;