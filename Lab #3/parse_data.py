import os
import re

import requests
from bs4 import BeautifulSoup

class Album:
    def __init__(self, name, url, tracks=None):
        self.__name = re.sub(r'[<>:"/\\|?*]', '', name)
        self.__url= url
        self.__tracks = tracks
        if tracks is None:
            self.__tracks = []

    def get_name(self):
        return self.__name

    def get_tracks(self):
        return self.__tracks

    def add_track(self, track):
        self.__tracks.append(track)


def get_urls(url, class_name, key_word):
    urls = []
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        container = soup.find('div', class_=class_name)
        if container:
            links = container.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href.find(key_word) != -1:
                    urls.append(href)
        else:
            print("Контейнер не найден.")
    else:
        print(f"Ошибка при загрузке страницы: {response.status_code}")
    response.close()
    return urls


def get_tracks_url(url, class_name, key_word):
    urls = []
    name = ""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.find('h1', class_="header_with_cover_art-primary_info-title header_with_cover_art-primary_info-title--white")
        if text:
            name = text.get_text()

        container = soup.find('div', class_=class_name)
        if container:
            links = container.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href.find(key_word) != -1:
                    urls.append(href)
        else:
            print("Контейнер не найден.")
    else:
        print(f"Ошибка при загрузке страницы: {response.status_code}")
    response.close()
    return urls, name

def get_track_info(track):
    lyrics_text = ""
    name = ""
    response = requests.get(track)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        name_span = soup.find('span', class_="SongHeader-desktop__HiddenMask-sc-9c2f20c9-11 cEehWv")
        if name_span:
            name = re.sub(r'[<>:"/\\|?*]', '', name_span.get_text())
        else:
            print("Контейнер с менем не найден.")

        lyrics_divs = soup.find_all('div', class_='Lyrics__Container-sc-926d9e10-1 fEHzCI')
        if lyrics_divs:
            for div in lyrics_divs:
                lyrics_text += div.get_text(separator="\n")  # separator добавляет переносы строк
        else:
            print("Контейнер с текстом не найден.")
    else:
        print(f"Ошибка при загрузке страницы: {response.status_code}")
    return name, lyrics_text

if __name__ == '__main__':
    dataset_path = "data"
    file_name = "file.txt"

    album_urls = get_urls("https://genius.com/artists/Korol-i-shut", "white_container", "albums")
    albums = []
    if album_urls is not None:
        tracks = []
        for url in album_urls:
            tracks, album_name = get_tracks_url(url, "column_layout u-top_margin", "lyrics")
            albums.append(Album(album_name, url, tracks))

    for i, album in enumerate(albums, start=1):
        album_path = os.path.join(dataset_path, album.get_name())
        os.makedirs(album_path, exist_ok=True)
        print(f"[{i}/{len(albums)}] - {album.get_name()}")

        for j, track in enumerate(album.get_tracks(), start=1):
            name, lyrics_text = get_track_info(track)
            lyrics_text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,!?\[\]]", "", lyrics_text)
            lyrics_text = re.sub(r"[\[]", "\n[", lyrics_text)

            filename = os.path.join(album_path, f"{name}.txt")
            with open(filename, "w", encoding='utf-8') as file:
                file.writelines(lyrics_text[1:])

            with open(os.path.join(dataset_path, file_name), "a", encoding='utf-8') as file:
                file.writelines(lyrics_text)

            print(f"    [{j}/{len(album.get_tracks())}] - {name}")




