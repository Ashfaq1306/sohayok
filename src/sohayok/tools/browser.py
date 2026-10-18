"""Open websites, search the web, play YouTube, and look things up."""

from __future__ import annotations

import html
import json
import re
import subprocess
import urllib.parse
import webbrowser

import httpx

from sohayok.lang import detect_language, normalize
from sohayok.tools.apps import app_paths_executable
from sohayok.tools.registry import ToolError, string, tool

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36"
)
SITES = {
    "youtube": "https://www.youtube.com",
    "ইউটিউব": "https://www.youtube.com",
    "google": "https://www.google.com",
    "গুগল": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "জিমেইল": "https://mail.google.com",
    "email": "https://mail.google.com",
    "ইমেইল": "https://mail.google.com",
    "facebook": "https://www.facebook.com",
    "ফেসবুক": "https://www.facebook.com",
    "messenger": "https://www.messenger.com",
    "মেসেঞ্জার": "https://www.messenger.com",
    "github": "https://github.com",
    "গিটহাব": "https://github.com",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "গুগল ড্রাইভ": "https://drive.google.com",
    "ড্রাইভ": "https://drive.google.com",
    "calendar": "https://calendar.google.com",
    "ক্যালেন্ডার": "https://calendar.google.com",
    "maps": "https://www.google.com/maps",
    "ম্যাপ": "https://www.google.com/maps",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "wikipedia": "https://www.wikipedia.org",
    "উইকিপিডিয়া": "https://bn.wikipedia.org",
    "linkedin": "https://www.linkedin.com",
    "লিংকডইন": "https://www.linkedin.com",
    "prothom alo": "https://www.prothomalo.com",
    "প্রথম আলো": "https://www.prothomalo.com",
    "news": "https://news.google.com",
    "খবর": "https://news.google.com/?hl=bn&gl=BD&ceid=BD:bn",
    "netflix": "https://www.netflix.com",
    "নেটফ্লিক্স": "https://www.netflix.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "instagram": "https://www.instagram.com",
    "ইনস্টাগ্রাম": "https://www.instagram.com",
    "translate": "https://translate.google.com",
    "অনুবাদ": "https://translate.google.com",
    "whatsapp web": "https://web.whatsapp.com",
}
SITES = {normalize(k): v for k, v in SITES.items()}  # lookups use normalized text
SEARCH = {
    "google": "https://www.google.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
    "bing": "https://www.bing.com/search?q={}",
    "wikipedia": "https://{lang}.wikipedia.org/w/index.php?search={}",
    "maps": "https://www.google.com/maps/search/{}",
    "github": "https://github.com/search?q={}",
    "images": "https://www.google.com/search?tbm=isch&q={}",
}


def chrome_path() -> str | None:
    return app_paths_executable("chrome.exe")


def open_in_browser(url: str) -> None:
    """Open a URL in Chrome if installed (it holds the user's signed-in sessions)."""
    chrome = chrome_path()
    if chrome:
        subprocess.Popen([chrome, url])
    else:
        webbrowser.open(url)


def site_url(target: str) -> str:
    key = normalize(target).removesuffix(".com").removesuffix(" ওয়েবসাইট").removesuffix(" website")
    if key in SITES:
        return SITES[key]
    if re.match(r"^https?://", target.strip()):
        return target.strip()
    if re.match(r"^[\w-]+(\.[\w-]+)+(/\S*)?$", target.strip()):
        return "https://" + target.strip()
    raise ToolError(f"unknown website {target!r}; give a URL or use web_search")


@tool(
    "Open a website or URL in Chrome. Accepts names like 'youtube', 'gmail', 'facebook', "
    "'github', 'google drive', 'প্রথম আলো', or a URL/domain.",
    {"site": string("Website name, domain, or URL")},
)
def open_website(site: str) -> str:
    url = site_url(site)
    open_in_browser(url)
    return f"opened {url}"


@tool(
    "Search the web and show the results in Chrome. Engines: google (default), youtube, "
    "bing, wikipedia, maps, github, images.",
    {"query": string("What to search for"), "engine": string("Search engine", list(SEARCH))},
    required=["query"],
)
def web_search(query: str, engine: str = "google") -> str:
    template = SEARCH.get(engine, SEARCH["google"])
    url = template.replace("{lang}", detect_language(query)).format(urllib.parse.quote_plus(query))
    open_in_browser(url)
    return f"searched {engine} for {query!r}"


def first_youtube_video(query: str, client: httpx.Client | None = None) -> tuple[str, str]:
    """Return (video_id, title) of the top YouTube search result for `query`."""
    client = client or httpx.Client(
        timeout=10, headers={"User-Agent": USER_AGENT, "Accept-Language": "en,bn;q=0.8"}
    )
    page = client.get("https://www.youtube.com/results", params={"search_query": query}).text
    for match in re.finditer(r'"videoRenderer":\{"videoId":"([\w-]{11})"', page):
        video_id = match.group(1)
        tail = page[match.end() : match.end() + 3000]
        title = re.search(r'"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"', tail)
        return video_id, json.loads(f'"{title.group(1)}"') if title else query
    raise ToolError(f"no YouTube video found for {query!r}")


@tool(
    "Play the top YouTube video for a search (a song, a video, a channel) in Chrome.",
    {"query": string("What to play, e.g. 'Ayub Bachchu Shei Tumi' or 'রবীন্দ্রসংগীত'")},
)
def play_youtube(query: str) -> str:
    try:
        video_id, title = first_youtube_video(query)
    except (ToolError, httpx.HTTPError):
        # No clear top video (only Shorts or mixes, or YouTube unreachable): show the results.
        web_search(query, "youtube")
        return f"no single top video for {query!r}; opened the YouTube search results instead"
    open_in_browser(f"https://www.youtube.com/watch?v={video_id}")
    return f"playing {title!r} on YouTube"


@tool(
    "Look up a short factual summary from Wikipedia (Bangla or English) to answer a question "
    "aloud. Use for 'who is', 'what is', 'কে', 'কী'.",
    {
        "topic": string("The person, place, or thing"),
        "language": string("Wikipedia language", ["bn", "en"]),
    },
    required=["topic"],
)
def wikipedia_summary(topic: str, language: str = "") -> str:
    language = language or detect_language(topic)
    headers = {"User-Agent": "sohayok/0.1 (desktop assistant)"}
    with httpx.Client(timeout=10, headers=headers) as client:
        found = (
            client.get(
                f"https://{language}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": topic,
                    "srlimit": 1,
                    "format": "json",
                },
            )
            .json()
            .get("query", {})
            .get("search", [])
        )
        if not found:
            raise ToolError(f"Wikipedia has no article for {topic!r}")
        title = found[0]["title"]
        summary = client.get(
            f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title.replace(" ", "_"))
        ).json()
    extract = summary.get("extract") or ""
    return f"{title}: {extract[:1200]}"


@tool(
    "Search the web and return the top results as text (titles and snippets), to answer a "
    "question that needs current information. Does not open the browser.",
    {"query": string("Search query")},
)
def web_lookup(query: str) -> list[dict[str, str]]:
    response = httpx.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    results = []
    pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
        re.S,
    )
    for href, title, snippet in pattern.findall(response.text)[:5]:
        clean = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()  # noqa: E731
        link = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        results.append({"title": clean(title), "snippet": clean(snippet), "url": link})
    if not results:
        raise ToolError("the web search returned no results")
    return results
