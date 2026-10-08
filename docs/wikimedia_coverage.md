# Đo độ phủ Wikidata và Wikipedia

`scripts/measure_wikimedia_coverage.py` đo trên toàn bộ `movies.csv` của MovieLens,
đồng thời đánh dấu membership trong `train.parquet`, `valid.parquet` và
`test.parquet`. Độ phủ dùng số **movieId duy nhất**, không dùng số ratings.

Các phạm vi đầu ra gồm catalog MovieLens gốc, hợp train/valid/test, và từng split.
Với dữ liệu hiện tại, catalog gốc có 27.278 phim; train có 11.508, valid 10.622,
test 11.386. Hợp ba split có 11.508 phim, vì mọi phim trong valid/test cũng xuất
hiện trong train. Script đọc cả ba split để kiểm tra điều này, không suy từ train.

## Chạy và tái lập

Không cần token hay dependency mới; sử dụng Polars và thư viện chuẩn Python.
Các path mặc định được neo theo repository root.

```bash
uv run python scripts/measure_wikimedia_coverage.py
uv run python scripts/measure_wikimedia_coverage.py --cache-only
```

Pilot ghi đầu ra riêng, trong khi raw cache có thể dùng lại:

```bash
uv run python scripts/measure_wikimedia_coverage.py --limit 50 \
  --output-dir artifacts/movie_content/wikimedia_coverage/pilot
```

`--mapping-only` chỉ đo Wikidata và sitelinks, chưa đo văn bản Wikipedia.
`--languages en` chỉ đo văn bản tiếng Anh. Mặc định đo cả tiếng Anh và tiếng Việt.
Mọi kết quả phải được đọc cùng stage, languages_requested và status counts;
partial/mapping-only không xác nhận độ phủ văn bản trên toàn catalog.

Script gửi tối đa một request/giây, dùng batch 100 IMDb ID cho WDQS và tối đa
50 bài cho Wikipedia. Có gzip, cache, retry, Retry-After và maxlag cho Action API.
Đầu ra hiện tại không thay đổi splits hay catalog metadata của crawler TMDB.

## Định nghĩa độ phủ

- **Wikidata exact unique:** IMDb ID khớp P345 của đúng một QID trên WDQS. Các
  trường hợp không tìm thấy, nhiều QID hoặc request lỗi được phân biệt. Đây là
  độ phủ qua phương pháp ghép ID này; tác phẩm có trên Wikimedia nhưng thiếu
  P345 vẫn có thể bị bỏ sót.
- **Wikidata metadata:** có truthy statement của property tương ứng. Cast gồm
  P161 hoặc P725 (voice actor). Các tỷ lệ này đo sự hiện diện của statement,
  chưa xác nhận chuẩn hóa đơn vị, đủ nhãn người/genre hoặc chất lượng giá trị.
- **Wikipedia sitelink:** QID có URL enwiki/viwiki. Đây mới là khả năng ghép bài.
- **Wikipedia verified article:** API trả về bài namespace 0 và wikibase_item
  khớp QID đã ghép; xử lý normalization và redirects. Bài franchise, bản khác,
  hoặc bài thiếu QID không được tính như nội dung đã xác minh của phim.
- **Lead:** phần trước heading đầu tiên có ít nhất 100 ký tự văn bản sau làm sạch.
- **Plot:** phần có heading Plot/Synopsis/Plot summary/Story/Premise hoặc các tên
  tiếng Việt tương ứng có ít nhất 100 ký tự; giữ subsection, dừng ở section ngang
  hoặc cao hơn. Threshold điều chỉnh qua `--min-chars`.
- **Lead or plot:** phim có ít nhất một trong hai loại văn bản đạt threshold.

Văn bản được đo từ revision wikitext mới nhất trả về bởi API, loại comment, ref,
template, table và markup. Parser không mở rộng template và không dựng HTML đầy
đủ. Đây là phép đo narrative theo quy tắc trên, chưa phải đánh giá chất lượng
ngữ nghĩa. Plot/Premise là nhóm heading được nhận diện, không đảm bảo tóm tắt
toàn bộ cốt truyện. Tác phẩm không thuộc loại phim truyện có thể có lead hữu ích
nhưng không có mục plot.

Wikidata description tiếng Anh thường là nhãn mô tả ngắn; nó được đo riêng,
không coi là plot. Wikipedia reception hoặc review của nhà phê bình nằm trong
bài không được coi là corpus user reviews.

## File và kiểm tra

- `artifacts/movie_content/wikimedia_coverage/coverage.parquet`: một dòng mỗi
  movieId, split membership, QID ứng viên, trạng thái, metadata presence, các
  URL, page ID/revision, số ký tự lead/plot và raw path.
- `artifacts/movie_content/wikimedia_coverage/manifest.json`: mẫu số từng scope,
  số lượng/tỷ lệ theo tiêu chí, trạng thái, cấu hình, fingerprint, phiên bản parser,
  hash script và request count.
- `data/raw/wikimedia_coverage/`: raw JSON gzip theo batch và timestamp.
- `*_raw_paths` liệt kê mọi response khi một batch cần continuation; raw response
  gốc được giữ nguyên để tái lập normalization và text metrics.
- `artifacts/movie_content/wikimedia_coverage/report.md`: báo cáo kết quả đã kiểm tra.
- `fetch_stats.json` trong cùng thư mục: thống kê lượt thu thập API ban đầu,
  tách khỏi lượt tái lập offline. Các count trước tái lập chỉ dùng đối soát thay
  đổi parser; số liệu cuối cùng nằm trong manifest/report.
- `validation.json`: đối chiếu ID, title, IMDb, membership các split, 110 số
  lượng/tỷ lệ và 811 raw responses; xác nhận 6 hash đầu vào giữ nguyên.
- `identity_issues.json`: 21 IMDb ID mơ hồ và 64 bài EN sai QID để xử lý riêng.

Các request thất bại/response thiếu dữ liệu vẫn là unknown, không biến thành
“phim không tồn tại”. HTTP 200 hoặc stage hoàn thành không tự đảm bảo 100% coverage;
đọc các trạng thái mismatch/unverified/missing và các số liệu riêng.

Kiểm tra offline bằng unittest đã có trong Python, không cài test runner:

```bash
uv run python -m unittest discover -s tests -p 'test_wikimedia_coverage.py' -v
uv run python -m compileall scripts/measure_wikimedia_coverage.py tests/test_wikimedia_coverage.py
```

Fixtures kiểm tra plot/subsection, nội dung rỗng, heading tiếng Việt, ref dùng lại
không làm mất đoạn giới thiệu/cốt truyện, IMDb mapping mơ hồ, redirect sang
franchise, response thiếu và hợp split có phim chỉ ở test.
Kết quả thực tế cần đối chiếu lại số dòng/ID với ZIP và splits, provenance với raw
cache, status counts với manifest, cùng hash các đầu vào trước/sau phép đo.

Nguồn phương thức: [Wikidata data access](https://www.wikidata.org/wiki/Wikidata:Data_access),
[MediaWiki Revisions API](https://www.mediawiki.org/wiki/API:Revisions),
[API etiquette](https://www.mediawiki.org/wiki/API:Etiquette).

Dữ liệu cấu trúc Wikidata là CC0. Văn bản Wikipedia giữ URL/revision và giấy phép
áp dụng, thường CC BY-SA; raw cache không đổi giấy phép của nội dung nguồn.
