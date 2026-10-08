# Khảo sát làm giàu ngữ nghĩa cho catalog phim

Ngày khảo sát: **06/10/2026**. Phạm vi: nội dung phim, review/comment và điểm đánh giá để bổ sung đặc trưng cho hệ gợi ý MovieLens của dự án.

**Khả thi về kỹ thuật. Metadata đã có độ phủ cao; review cần khảo sát riêng về độ phủ và quyền sử dụng.** Hướng nên thử trước là user tags của MovieLens và nội dung từ Wikimedia. Nguồn review có thể lấy qua API hoặc corpus nghiên cứu, nhưng quyền truy cập API không đồng nghĩa với quyền tạo dataset và dùng cho ML/AI.

Các số liệu dưới đây được đo từ Parquet, manifest, ZIP và raw cache đang có. Thông tin nguồn ngoài được đối chiếu với tài liệu chính thức; độ phủ review trên catalog của dự án chưa được đo.

## 1. Dữ liệu và khả năng hiện có trong dự án

Đọc `catalog.parquet` và đối chiếu `manifest.json`: **11.508 movieId duy nhất**, tương ứng catalog train sau 10-core. Toàn bộ MovieLens gốc có **27.278 phim**; độ phủ dưới đây chỉ áp dụng cho catalog train.

| Dữ liệu | Số phim trong catalog train | Độ phủ |
| --- | ---: | ---: |
| Overview đang được chọn | 11.483 | 99,78% |
| Overview từ TMDB | 11.401 | 99,07% |
| Mô tả bổ sung từ Wikipedia | 82 | 0,71% |
| Keywords đã thu thập | 10.917 | 94,86% |
| Diễn viên | 11.433 | 99,35% |
| Đạo diễn | 11.473 | 99,70% |
| User tags trong MovieLens ZIP | 10.881 | 94,55% |
| Tag Genome trong MovieLens ZIP | 9.844 | 85,54% |

Các mô tả hiện có đều được đánh dấu tiếng Anh. Độ dài overview trung vị là **258 ký tự**; phân vị 10%/90% là 123/451 ký tự. Đây chủ yếu là mô tả ngắn, nên cốt truyện dài và nhận xét về phong cách phim vẫn có thể bổ sung thông tin.

ZIP chứa **465.564 lượt gắn tag**, trong đó **437.350 lượt** thuộc catalog train; có **31.288 tag phân biệt** sau strip và casefold. Tag Genome có **1.128 chiều**, 10.381 phim ở nguồn gốc, 9.844 phim giao với catalog train.

Các số liệu tags/Genome này tính trên nguồn đầy đủ, **chưa lọc theo cặp user–movie của train**. Độ phủ của một phiên bản tránh sử dụng tín hiệu từ validation/test sẽ cần đo lại.

Crawler hiện tại:

- Lấy TMDB movie details cùng credits và keywords; fallback Wikidata/Wikipedia.
- Có kiểm tra IMDb ID, cache, retry/backoff, checkpoint và provenance.
- Chưa gọi reviews endpoint, chưa tạo embeddings, chưa đưa user tags/Genome vào catalog.
- Raw TMDB details có `vote_average` và `vote_count`, nhưng catalog chuẩn hóa chưa lưu hai trường này. Có thể bổ sung từ response đã xác minh mà không cần gọi lại API, nếu phạm vi sử dụng dữ liệu cho phép.
- Có 25 phim thiếu overview, 79 mapping chỉ ra tác phẩm TV và hai mapping chưa xác minh. Việc mở rộng review phải giữ các trạng thái này để tránh ghép nhầm tác phẩm.

**82 mô tả Wikipedia là kết quả fallback hiện tại; con số này không đo độ phủ tiềm năng của Wikipedia trên toàn catalog.**

Phép đo độc lập trên toàn bộ 27.278 phim và cả train/valid/test đã hoàn tất trong
[báo cáo độ phủ Wikimedia](../artifacts/movie_content/wikimedia_coverage/report.md).
Trên catalog gốc, Wikidata khớp duy nhất 96,29%; Wikipedia EN có giới thiệu hoặc
plot/premise từ 100 ký tự cho 90,55% phim, phần plot/premise phủ 82,17%.
Trên hợp train/valid/test, các tỷ lệ tương ứng là 99,34%, 98,37% và 93,80%.

Nguồn cục bộ: [crawler](../scripts/crawl_movie_metadata.py), [hướng dẫn metadata](movie_content.md), [manifest](../data/processed/movie_content/manifest.json), [báo cáo crawl trước](../artifacts/movie_content/report.md), [README MovieLens](../data/raw/ml-20m/README.txt).

## 2. So sánh các nguồn

Mức phù hợp bên dưới là đánh giá cho mục tiêu làm giàu ngữ nghĩa của dự án, dựa trên tính năng công bố và điều kiện sử dụng; chưa phải benchmark coverage.

| Nguồn | Nội dung/metadata | Review và đánh giá | Đường lấy dữ liệu và mức phù hợp |
| --- | --- | --- | --- |
| **MovieLens tags/Genome** | Tags tự do; relevance của các thuộc tính phim | Không có nguyên văn review trong ML-20M | Đã có trong ZIP, join trực tiếp movieId. Ưu tiên nghiên cứu; xử lý nguy cơ dùng tín hiệu từ tập đánh giá. [GroupLens](https://grouplens.org/datasets/movielens/20m/) |
| **Wikidata** | Genre, người tham gia, quốc gia, ngôn ngữ, quan hệ và ID | Không phải kho user review | API/query/bulk dump; dữ liệu cấu trúc CC0. Phù hợp cho graph ngữ nghĩa và cầu nối ID. [Data access](https://www.wikidata.org/wiki/Wikidata:Data_access) |
| **Wikipedia** | Plot, synopsis, giới thiệu, mục reception | Reception là bài tổng hợp có dẫn nguồn, không tương đương từng user review | API hoặc dump; văn bản thường CC BY-SA 4.0, giữ nguồn và phiên bản. Phù hợp cho mô tả sâu. [Điều khoản Wikimedia](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) |
| **CMU Movie Summary Corpus** | 42.306 plot summaries; metadata/nhân vật | Không có user review | Bulk corpus CC BY-SA; snapshot Wikipedia năm 2012. Dùng Wikipedia/Freebase ID làm cầu nối; cần đo overlap. [Corpus](https://www.cs.cmu.edu/~ark/personas/), [README](https://www.cs.cmu.edu/~ark/personas/data/README.txt) |
| **TMDB** | Overview, tagline, keywords, credits, collection | API có user review, rating của reviewer và điểm tổng hợp | Dễ mở rộng crawler hiện tại. Quyền sử dụng cho ML/AI cần thỏa thuận phù hợp; chưa đo review coverage. [Reviews API](https://developer.themoviedb.org/reference/movie-reviews), [API terms](https://www.themoviedb.org/api-terms-of-use?language=en-CA) |
| **OMDb API** | Plot short/full, metadata theo IMDb ID | Điểm tổng hợp; không có endpoint corpus user review trong API công bố | Nguồn phụ cho nghiên cứu phi thương mại; trang công bố CC BY-NC 4.0 và quota miễn phí 1.000/ngày. [API](https://www.omdbapi.com/), [quota](https://www.omdbapi.com/apikey.aspx) |
| **IMDb official datasets** | Title, genre, runtime, cast/crew, ID | Có averageRating và numVotes; bộ TSV miễn phí không có review text/plot | Bulk gzip TSV, cập nhật hằng ngày; dùng theo điều kiện phi thương mại. Không crawl website trực tiếp theo điều khoản hiện có. [Datasets](https://www.imdb.com/interfaces/?opt_id=undefined), [quy định sử dụng](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX) |
| **Stanford aclImdb** | Tập trung vào review text | 50.000 review có nhãn sentiment và 50.000 review không nhãn | Corpus benchmark có sẵn, phù hợp thử phương pháp NLP. Cần kiểm tra khóa ghép phim trong gói gốc và quyền sử dụng cụ thể. [Nguồn Stanford](https://ai.stanford.edu/~amaas/data/sentiment/) |
| **IMDb Spoiler Dataset** | Có metadata phim | 573.913 review, 1.572 phim; có movie_id, ngày, rating và spoiler label | Có thể thử pipeline aspect/spoiler theo phim; 1.572 phim chỉ bằng tối đa 13,66% catalog hiện tại ngay cả khi tất cả đều khớp. Quyền tái sử dụng cần xác minh. [Tác giả](https://rishabhmisra.github.io/publications/), [mô tả dataset](https://rishabhmisra.github.io/IMDBSpoilerDataset.pdf) |
| **Trakt** | Catalog và ID liên kết | Có API movie comments | Có đường API, nhưng chính sách 22/09/2026 cấm bulk harvesting/mirroring catalog và dữ liệu cộng đồng. Cần thỏa thuận cho mục tiêu tạo corpus. [API reference](https://trakt.docs.apiary.io/), [API Use Policy](https://developer.trakt.tv/docs/api-use-policy) |
| **Letterboxd** | Catalog và liên kết phim | User review/rating | API riêng cấp cho đối tác; điều khoản hạn chế thu thập tự động. Phù hợp khi được cấp quyền. [API access](https://letterboxd.zendesk.com/hc/en-us/articles/15269070369551-Do-you-have-mobile-apps-or-an-API), [terms](https://letterboxd.com/legal/terms-of-use/) |
| **Rotten Tomatoes** | Catalog và thông tin đánh giá | Critic/audience scores; phạm vi review tùy gói | API/feed qua quy trình cấp phép. Cần xác nhận quyền full text và ML/AI trong hợp đồng. [Licensing](https://www.rottentomatoes.com/help_desk/licensing) |

Giấy phép MovieLens cho phép nghiên cứu theo các điều kiện trong README; tái phân phối và sử dụng thương mại cần quyền riêng. Genome được tính từ tags, ratings và review, nên không coi là đặc trưng độc lập với tín hiệu đánh giá.

## 3. Điểm quyết định đối với việc dùng dữ liệu cho mô hình

**TMDB:** FAQ về API miễn phí phi thương mại và attribution chưa đủ để kết luận có quyền dùng cho mô hình. Điều khoản API được lập chỉ mục có hạn chế ứng dụng ML/AI và cache quá sáu tháng; điều khoản website yêu cầu cho phép bằng văn bản đối với training/validation. Cần xác nhận phạm vi cả embedding inference, huấn luyện recommender và chia sẻ corpus. [FAQ](https://developer.themoviedb.org/docs/faq), [API terms](https://www.themoviedb.org/api-terms-of-use?language=en-CA), [website terms](https://www.themoviedb.org/terms-of-use?language=ar-SA).

Trong lần khảo sát này, trang TMDB terms không tải trực tiếp được; nội dung điều khoản trên được đọc qua bản lập chỉ mục của trang chính thức. Cần kiểm tra bản hiện hành trực tiếp trước khi quyết định sử dụng.

Hướng dẫn metadata hiện tại chủ yếu mô tả attribution. Vì vậy, kết luận “dữ liệu sẵn sàng làm đặc trưng” trong báo cáo crawl cũ mới phản ánh tình trạng kỹ thuật, chưa xác nhận giấy phép ML/AI.

**Corpus có sẵn:** nhãn giấy phép trên Kaggle/Hugging Face hoặc giấy phép của code tải dữ liệu không tự xác nhận quyền đối với văn bản review gốc. Với aclImdb/IMDb Spoiler, cần kiểm tra điều kiện của đúng bản phát hành. Đây là ứng viên benchmark, chưa đủ cơ sở để coi là nguồn review thương mại phủ toàn catalog.

**Nguồn mở:** nên giữ Wikidata và Wikipedia ở các trường nguồn riêng. CC0 của Wikidata không áp dụng cho văn bản Wikipedia. Với CMU, link giấy phép trên trang corpus là CC BY-SA 3.0 US; với Wikipedia hiện hành, giữ giấy phép và attribution của từng bài/phiên bản.

## 4. Phương thức thu thập

| Phương thức | Điểm mạnh | Việc cần giải quyết |
| --- | --- | --- |
| **Đọc dữ liệu/bulk corpus** | Snapshot tái lập được; ít phụ thuộc DOM | Version, mapping, overlap, giấy phép |
| **API chính thức** | Schema rõ; ID, phân trang và lỗi có cấu trúc | API key, quota, quyền tạo corpus/ML, cache TTL |
| **HTML/JSON-LD ở nguồn cho phép** | Có thể lấy thông tin chưa có API | Parser dễ hỏng; kiểm tra điều khoản và robots của đúng nguồn |
| **Browser automation ở nguồn cho phép** | Xử lý review tải động | Tốn tài nguyên, dễ đổi UI; giới hạn truy cập và quyền sử dụng vẫn áp dụng |
| **Feed/hợp đồng cấp phép** | Có thể xác nhận full text, bulk access và quyền ML | Chi phí và phạm vi cần hỏi nhà cung cấp |
| **Review do hệ thống tự thu nhận** | Gắn trực tiếp movieId; có thể lấy tiếng Việt | Cần điều khoản/đồng ý phù hợp và thời gian tích lũy |

Nếu cần ngữ nghĩa tiếng Việt, có thể lấy sitelink Wikipedia tiếng Việt khi tồn tại, giữ song song văn bản gốc và bản dịch, hoặc thu nhận review của người dùng hệ thống. Phép đo toàn catalog đã xác minh 1.959 bài tiếng Việt; 1.949 phim có giới thiệu hoặc cốt truyện từ 100 ký tự (7,14% catalog gốc). Trên hợp train/valid/test, nội dung tiếng Việt phủ 1.512/11.508 phim (13,14%). Xem [báo cáo Wikimedia](../artifacts/movie_content/wikimedia_coverage/report.md).

## 5. Thiết kế ghép ID và dữ liệu review

Dùng chuỗi `movieId → imdb_id / tmdb_id → source_movie_id`. Giữ IMDb ID ở dạng `tt...` có số 0 đầu. Với Wikipedia/CMU, dùng Wikidata làm cầu nối IMDb ID đến bài Wikipedia/Freebase ID; xử lý bài đổi tên hoặc merge. Nếu phải dò theo title, đối chiếu thêm năm, media type và người tham gia; lưu trạng thái ambiguous thay vì ép ghép.

Cần tách ba loại thông tin:

- **Nội dung phim:** plot, chủ đề, bối cảnh, genre, người tham gia.
- **Phản hồi khán giả:** nhịp phim, diễn xuất, cảm xúc, hình ảnh, âm nhạc, tính dễ xem.
- **Đánh giá tổng hợp:** score, số vote, nguồn và ngày snapshot.

Điểm tổng hợp phản ánh sự yêu thích; review cung cấp thêm lý do và thuộc tính trải nghiệm. Hai loại này phục vụ các mục tiêu đặc trưng khác nhau.

Đề xuất `reviews.parquet` có một dòng/review với `movieId`, `source`, `source_movie_id`, `review_id`, `source_url`, `text`, `text_language`, `rating`, `rating_scale`, `created_at`, `updated_at`, `fetched_at`, `spoiler_flag`, `license`, `usage_scope`, `raw_path`. Các trường nguồn không cung cấp để null; request language và ngôn ngữ phát hiện trong text nên được lưu riêng. Hồ sơ cá nhân của tác giả không cần cho đặc trưng phim.

TMDB có `GET /3/movie/{movie_id}/reviews` với `language` và `page`. Phản hồi có `total_pages`, `total_results`; từng review có content, ID, URL, thời điểm và `author_details.rating`. [OpenAPI chính thức](https://developer.themoviedb.org/openapi/tmdb-api.json).

Khử trùng theo source/review_id và text hash; lọc HTML, spam và text quá ngắn; giữ reviewer rating riêng với aggregate rating. Lấy mẫu theo rating và thời gian nếu dữ liệu cho phép, thay vì chỉ lấy review được thích nhất. Một phim ít review cần quality/count/missing flags để mô hình nhận biết độ tin cậy.

## 6. Dùng ngữ nghĩa trong hệ gợi ý

LightGCN hiện tại dùng embedding ID và graph tương tác; catalog mới sẽ cần một bước tích hợp riêng. [Model](../lightgcn/model.py), [ID mapping](../lightgcn/data.py).

Nên bắt đầu bằng các thử nghiệm tách được đóng góp:

1. **User tags:** chuẩn hóa, đếm số người gắn tag, tạo vector tag hoặc văn bản tag.
2. **Nội dung nguồn mở:** plot + thể loại + thông tin cấu trúc; giữ provenance của từng phần.
3. **Review có quyền dùng:** tạo vector riêng và các thuộc tính như pacing, acting, visuals, mood; giữ bằng chứng nguồn cho thuộc tính được trích xuất.
4. **Kết hợp với CF:** tạo hồ sơ ngữ nghĩa người dùng từ các phim trong train, rồi trộn score ngữ nghĩa với score LightGCN đã chuẩn hóa. Chọn trọng số trên validation.
5. **Genome:** thử ở nhánh riêng, công bố nguồn tín hiệu và khả năng phụ thuộc vào ratings/reviews.

Đánh giá Recall@20, NDCG@20, catalog coverage và kết quả theo nhóm popular/long-tail. Có thể dùng kiểm tra nearest neighbors thủ công để xem đặc trưng phản ánh đúng nội dung.

Split hiện tại là random theo user, không theo thời gian. Metadata/review lấy năm 2026 là snapshot sau giai đoạn ratings MovieLens 1995–2015. Thí nghiệm cần công bố bối cảnh này; một đánh giá mô phỏng tương lai cần cutoff theo thời gian riêng.

Đối với tags, bản thận trọng chỉ dùng tags của cặp user–movie thuộc train để dựng đặc trưng. Hồ sơ người dùng không lấy các phim validation/test. Review/score cộng đồng và Genome cần đánh giá riêng ảnh hưởng của thông tin đến sau thời điểm dự đoán.

## 7. Pilot để ra quyết định

Đây là đề xuất công việc tiếp theo, chưa thực hiện thu thập review.

**Giai đoạn A — nguồn đã có và nguồn mở**

- Xuất thống kê/đặc trưng tags từ ZIP với quy tắc train đã chọn.
- Lấy mẫu Wikipedia/CMU trên 300 phim: 100 popular, 100 trung bình, 100 long-tail, xác định bằng số tương tác train.
- Đo mapping hợp lệ, plot coverage, độ dài, ngôn ngữ, số request và chất lượng trích xuất. Bổ sung một tập nhỏ trường hợp thiếu ID/TV/ambiguous để kiểm tra mapping.

**Giai đoạn B — review sau khi xác định quyền sử dụng**

- Chọn corpus nghiên cứu hoặc API/feed được cho phép.
- Với API, dùng cùng mẫu 300 phim, tối đa 20 review/phim cho pilot; lưu cả phim không có review.
- Đo tỷ lệ phim có ít nhất 1 và 5 review hợp lệ, phân vị số review, độ phủ theo popularity, độ dài, rating còn thiếu, spoiler, chi phí và lỗi.
- Chỉ mở rộng khi review tạo thêm thông tin hữu ích, mapping đúng và phạm vi sử dụng phù hợp.

Ước lượng năng lực, **không phải kết quả đo**:

- Một lượt đầu reviews API/phim trên 11.508 phim là khoảng 11.508 request. Ở cấu hình 5 request/giây của crawler hiện tại, riêng rate budget là khoảng 38,4 phút; phân trang, retry và latency tăng thời gian. TMDB yêu cầu xử lý HTTP 429; giới hạn nhà cung cấp có thể đổi. [Rate limiting](https://developer.themoviedb.org/docs/rate-limiting).
- Nếu thu được trung bình 20 review/phim, có khoảng 230.160 review. Giả định 3 KiB text/review, riêng text khoảng 674 MiB trước nén, chưa tính raw JSON.
- Vector float32 384 chiều cho 11.508 phim chiếm khoảng 16,9 MiB mỗi loại vector. Giá trị của review nên được kiểm chứng trước khi tăng số lượng text.

**Đề xuất cho dự án:** ưu tiên user tags và nhánh nội dung Wikimedia/CMU có provenance rõ; giữ Genome ở thử nghiệm riêng. Với review, xác minh quyền của corpus/nhà cung cấp rồi đo coverage trên pilot. Chưa có cơ sở để kỳ vọng review miễn phí phủ gần toàn bộ catalog như metadata hiện tại.
