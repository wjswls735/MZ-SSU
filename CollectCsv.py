from korea_news_crawler.articlecrawler import ArticleCrawler

Crawler = ArticleCrawler()  
#Crawler.set_category("정치", "IT과학", "경제", "사회", "생활문화","세계", "오피니언")  
Crawler.set_category("정치", "IT과학", "경제")  
Crawler.set_date_range(2017, 1, 2017, 12)  
Crawler.start()
