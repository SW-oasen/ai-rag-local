# Lokaler RAG-Assistent

Ein lokaler Retrieval-Augmented-Generation-Assistent für eigene Dokumente. Das Projekt lädt lokale Dateien, extrahiert Text, zerlegt ihn in sinnvolle Chunks, erzeugt Embeddings über Ollama, speichert diese in ChromaDB und beantwortet Fragen mit einem lokalen LLM inklusive Quellenangaben.

Die README beschreibt vor allem die lokale Weboberfläche. Technische Details, Architekturentscheidungen, Hardware-Annahmen und die vollständige CLI-Referenz stehen in `project_context.md`.

## Funktionen

- Lokale Dokumentbibliothek für Markdown-, Text- und PDF-Dateien
- Optionales OCR für gescannte PDFs über Tesseract
- Semantische Suche über lokale Embeddings
- Frage-Antwort-Funktion mit lokalem LLM
- Quellenanzeige für Antworten und gefundene Chunks
- Dokumentzusammenfassungen mit Cache und Export
- Textextraktion mit Vorschau und `.txt`-Export
- Lokale Konfiguration von Dokumentpfaden und Speicherorten

## Architektur

```text
Lokale Dokumente laden
    |
    v
Chunking, Embedding -> ChromDB
    |
    v
Text exahieren  -> Auch gescannte PDFs und Bilder mit OCR
    |
    v
Fragen - Zusammenfassen
    |
    v
Lokale LLMs
    |
    v
Antworten mit Quellenangaben
```

## Voraussetzungen

- Python 3.10 oder neuer
- Ollama
- Ein lokal installiertes Embedding-Modell
- Ein lokal installiertes Generationsmodell
- Optional: Tesseract OCR für gescannte PDFs

Empfohlene Modelle:

- Embeddings: `bge-m3` für Deutsch und Englisch, alternativ `nomic-embed-text`
- Antworten: `qwen3:8b` für lokale Nutzung, `qwen3-coder:30b` bei stärkerer Hardware

Modelle installieren:

```powershell
ollama pull bge-m3
ollama pull qwen3:8b
```

## Installation

Virtuelle Umgebung erstellen und Abhängigkeiten installieren:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

Falls Windows die Zertifikatsprüfung für PyPI blockiert:

```powershell
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .[dev]
```

OCR-Abhängigkeiten installieren:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[ocr]
winget install UB-Mannheim.TesseractOCR
```

Nach der Tesseract-Installation ein neues Terminal öffnen und prüfen:

```powershell
tesseract --version
tesseract --list-langs
```

Die OCR-Dropdownliste in der Web UI bietet Englisch (`eng`), Deutsch (`deu`), Englisch+Deutsch (`eng+deu`), Französisch (`fra`), vereinfachtes Chinesisch (`chi_sim`) und traditionelles Chinesisch (`chi_tra`). Die jeweilige Sprache funktioniert nur, wenn das passende Tesseract-Sprachpaket lokal installiert ist.

## Web UI Starten

Lokale Browseroberfläche starten:

```powershell
.\.venv\Scripts\rag-assistant-ui.exe
```

Danach im Browser öffnen:

```text
http://127.0.0.1:8765
```

Die Web UI läuft standardmäßig nur lokal unter `127.0.0.1` und nutzt die in `config.py` definierten Modelle und Speicherpfade, sofern beim Start keine anderen Optionen gesetzt werden.

## Menüpunkt: Overview

`Overview` ist die Startseite der Web UI. Sie zeigt den aktuellen Zustand der lokalen Dokumentbibliothek.

Sichtbar sind:

- Anzahl der indexierten Dokumente
- Anzahl der gespeicherten Chunks
- erkannte Seitenanzahl, falls verfügbar
- Anzahl konfigurierter Dokumentpfade
- unterstützte Dateitypen
- aktueller Vector-Store-Pfad
- aktueller Library-Store-Pfad
- Tabelle der indexierten Quellen mit Dateityp, Chunk-Anzahl, Seiten und Pfad

Typische Nutzung:

1. Nach dem Start prüfen, ob bereits Quellen indexiert sind.
2. Über die Quick Links zu Fragen, Zusammenfassungen, Textextraktion oder Konfiguration wechseln.
3. Nach einer Ingestion kontrollieren, ob neue Dokumente im Quellenbereich auftauchen.

## Menüpunkt: Ask

`Ask` ist der Bereich für semantische Suche und Frage-Antwort-Nutzung.

Funktionen:

- Frage an alle indexierten Quellen stellen
- Frage auf eine einzelne Quelle begrenzen
- Anzahl der abgerufenen Chunks über `Top K` steuern
- Nur Retrieval ausführen, ohne eine LLM-Antwort zu erzeugen
- Antwort mit lokalem LLM erzeugen
- Quellen der Antwort anzeigen
- Gefundene Chunks mit Score und Textvorschau prüfen

Typischer Workflow:

1. Frage eingeben.
2. Optional eine Quelle auswählen.
3. `Top K` passend wählen, zum Beispiel `4` oder `5`.
4. Erst `Retrieve` nutzen, um die gefundenen Chunks zu prüfen.
5. Danach `Ask` nutzen, um aus den Chunks eine Antwort erzeugen zu lassen.

Hinweis: Wenn die gefundenen Chunks nicht passen, sollte zuerst Retrieval geprüft werden, bevor Prompts oder Modelle angepasst werden.

## Menüpunkt: Summarize

`Summarize` erstellt und verwaltet Zusammenfassungen für indexierte Quellen.

Funktionen:

- Eine indexierte Quelle auswählen
- Bereits gecachte Zusammenfassung anzeigen
- Zusammenfassung neu erzeugen oder aktualisieren
- Zusammenfassung als Markdown exportieren
- Zusammenfassung als Textdatei exportieren
- Anzahl der Teilsummaries nachvollziehen

Typischer Workflow:

1. Dokument auswählen.
2. `View Cached Summary` prüfen, falls bereits eine Zusammenfassung existiert.
3. `Generate / Update Summary` nutzen, um eine neue Zusammenfassung zu erstellen.
4. Ergebnis über `Export .md` oder `Export .txt` speichern.

Die Zusammenfassung nutzt einen eigenen Ablauf über alle Chunks der Quelle. Sie ist nicht nur eine normale Top-k-Frage.

## Menüpunkt: Extract Text

`Extract Text` dient zur Textprüfung vor der Indexierung oder zur separaten Textextraktion.

Funktionen:

- Datei- oder Ordnerpfad eingeben
- Text aus unterstützten Dateitypen extrahieren
- Optional OCR aktivieren
- OCR-Sprache über eine Dropdownliste wählen, zum Beispiel `eng`, `deu`, `fra`, `chi_sim` oder `chi_tra`
- OCR-Skalierung und Page-Segmentation-Modus konfigurieren
- Vorverarbeitung und Textbereinigung aktivieren oder deaktivieren
- Extrahierten Text in der Web UI prüfen
- Extrahierten Text als `.txt` exportieren; der Dateiname wird aus der Quelle abgeleitet

Typischer Workflow:

1. Pfad zu einer Datei oder einem Ordner eingeben.
2. OCR ist standardmäßig aktiviert, wird bei PDFs aber nur genutzt, wenn keine normale Textextraktion möglich ist.
3. Bei gescannten PDFs die OCR-Sprache passend setzen, zum Beispiel `eng`, `deu`, `fra`, `chi_sim` oder `chi_tra`.
4. `Extract Text` ausführen und die Textqualität prüfen.
5. Falls kein Text gefunden wird, OCR-Sprache, installierte Tesseract-Sprachpakete, Skalierung oder `PSM` prüfen.
6. Text optional über `Export .txt` speichern.

## Menüpunkt: Configuration

`Configuration` verwaltet lokale Dokumentpfade, Indexing-Aktionen und gecachte Zusammenfassungen.

Funktionen:

- Dokumentpfad hinzufügen
- Konfigurierte Pfade anzeigen
- Einzelnen Pfad indexieren
- Pfad wieder entfernen
- Zusammenfassung für eine Quelle erzeugen oder aktualisieren
- Gecachte Zusammenfassung entfernen
- Einzelne indexierte Quelle aus dem Vector Store löschen
- Gesamten Vector Store zurücksetzen
- Aktuelle Speicherorte für Vector Store und Library Store anzeigen

Typischer Workflow:

1. Einen lokalen Dokumentpfad hinzufügen, zum Beispiel `data/raw` oder einen absoluten Pfad zu einem PDF.
2. Den Pfad über `Ingest` indexieren.
3. Zur `Overview` wechseln und prüfen, ob die Quellen sichtbar sind.
4. Optional in `Configuration` eine Zusammenfassung für eine Quelle cachen.
5. Bei Bedarf eine einzelne Quelle löschen oder den Vector Store zurücksetzen.
6. Danach in `Ask` Fragen stellen oder in `Summarize` Zusammenfassungen ansehen.

## Lokale Speicherung

Der ChromaDB-Index wird standardmäßig im Projektordner gespeichert:

```powershell
.\vector_store
```

Der UI-Store für konfigurierte Pfade und gecachte Zusammenfassungen liegt standardmäßig unter:

```text
data/processed/web_library.json
```

Der Ordner `vector_store` ist in `.gitignore` ausgeschlossen und kann lokal jederzeit neu aufgebaut werden. Wenn ChromaDB auf einem Projektlaufwerk einen SQLite-Disk-I/O-Fehler meldet, ist ein Vector-Store-Pfad auf einem anderen lokalen Laufwerk oft die einfachste Lösung. Details und CLI-Optionen stehen in `project_context.md`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

Die Tests prüfen unter anderem Dokumentladen, Chunking, Retrieval, Prompt-Erstellung, Pipeline-Verhalten, Zusammenfassung und CLI-Ausgabeformatierung.

## Grenzen

- Die Qualität hängt stark vom lokalen Modell ab.
- PDF-Extraktion kann bei komplexem Layout, Tabellen und Scans unvollständig sein.
- OCR ist langsamer und benötigt eine lokale Tesseract-Installation.
- Semantisches Retrieval ist nicht perfekt; relevante Chunks sollten in `Ask` zuerst mit `Retrieve` geprüft werden.
- Antworten sollen aus dem bereitgestellten Kontext entstehen, können aber bei schwachem Modell oder unpassendem Kontext trotzdem Fehler enthalten.

## Weitere Informationen

- Technische Architektur: `project_context.md`
- Vollständige CLI-Referenz: `project_context.md`
- Beispiel-Evaluationen: `examples/`
- Lokale Web-App-Implementierung: `src/rag_assistant/web_app.py`


---

Autor: Yuchuan Liu
Letzte Aktualisierung: 2026-06-26
