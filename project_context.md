# Projektkontext: Lokaler RAG-Assistent

Dieses Dokument beschreibt die technischen Entscheidungen, Hardware-Annahmen, Implementierungsdetails, Entwicklungsleitlinien und CLI-Referenz des Projekts. Die README ist dagegen auf Überblick, Installation und die Nutzung der Web UI ausgerichtet.

## 1. Zielbild

Das Projekt baut einen lokalen dokumentenbasierten Frage-Antwort-Assistenten mit Retrieval-Augmented Generation. Der Assistent soll lokale Dokumente einlesen, in Chunks zerlegen, lokal einbetten, semantisch durchsuchen und Antworten mit Quellenhinweisen über ein lokales LLM erzeugen.

Wichtige Ziele:

- Lokale Ausführung ohne verpflichtende externe LLM-APIs
- Verständliche, modulare RAG-Architektur
- Portfoliofähige Dokumentation und Codebasis
- Gute Debuggbarkeit für Retrieval, Prompting und Quellenzuordnung
- Erweiterbarkeit in Richtung lokaler AI-Agenten

## 2. Hardware- und Laufzeitannahmen

Das Projekt ist für einen lokalen AI-PC beziehungsweise eine leistungsfähige Workstation gedacht. Es soll aber so implementiert bleiben, dass auch kleinere Modelle nutzbar sind.

Annahmen:

- Betriebssystem: Windows, Entwicklung in PowerShell und VS Code
- Python: Version 3.10 oder neuer
- Lokaler LLM-Runtime: Ollama
- Vektorindex: ChromaDB mit lokaler Persistenz
- Standard-UI: lokale HTTP-Oberfläche ohne externes Webframework
- CLI: installierbare Entry Points über `pyproject.toml`

Modellstrategie:

- `qwen3:8b` als praktikables Standardmodell für lokale Antworten
- `qwen3-coder:30b` als qualitativ stärkeres Modell, wenn genügend VRAM/RAM vorhanden ist
- `bge-m3` als bevorzugtes Embedding-Modell für deutsche und englische Dokumente
- `nomic-embed-text` als schnelle Alternative

Leistungsaspekte:

- Größere Generationsmodelle verbessern Antwortqualität, erhöhen aber Latenz und Speicherbedarf.
- Embedding-Batches sollten bei knapper Hardware reduziert werden, zum Beispiel `--embedding-batch-size 4`.
- OCR ist rechenintensiv und sollte nur für PDFs ohne extrahierbaren Text aktiviert werden.
- Der Vector Store liegt standardmäßig im temporären lokalen Benutzerverzeichnis, nicht zwingend im Repository.

## 3. Aktueller Implementierungsstand

Vorhanden:

- Dokumentladen für Markdown, Text und PDFs
- Optionales OCR für PDF-Seiten ohne verwertbaren Text
- Text-Splitting mit Chunk-Größe, Überlappung und Boundary-Verbesserungen
- Metadaten je Chunk, darunter Quelle, Dateiname, Chunk-Index, Seitenzahl und Zeichenbereiche
- Ollama-Embedding-Provider
- ChromaDB-basierter lokaler Vector Store
- Retriever für semantische Top-k-Suche
- Prompt Builder mit Quellenkontext
- Ollama-LLM-Client
- RAG-Pipeline mit Antwort, Quellen, Chunks, Modell und Prompt
- Dedizierter Map-Reduce-Summarizer für vollständige Dokumentzusammenfassung
- Retrieval-Evaluation über Markdown-Beispieldateien
- CLI mit `ingest`, `sources`, `chunks`, `retrieve`, `ask`, `summarize` und `eval`
- Lokale Browser-UI mit `Overview`, `Ask`, `Summarize`, `Extract Text` und `Configuration`
- Tests für Kernmodule und Ausgabeformatierung

## 4. Technischer Stack

Kerntechnologien:

- Python
- Ollama Python Client
- ChromaDB
- pypdf
- pytest
- Standardbibliothek für CLI und lokale Weboberfläche

Optionale OCR-Technologien:

- Tesseract OCR
- pytesseract
- Pillow
- pypdfium2

Bewusste Entscheidungen:

- Kein schweres RAG-Framework als Pflichtabhängigkeit
- Keine Cloud-API als Standardpfad
- Keine Datenbankserver-Abhängigkeit
- UI bewusst einfach und lokal, damit die RAG-Pipeline im Mittelpunkt bleibt

## 5. Architektur

### Ingestion

```text
Datei oder Ordner
    |
    v
Document Loader
    |
    v
Document-Objekte mit Text und Metadaten
    |
    v
Text Splitter
    |
    v
TextChunk-Objekte
    |
    v
Ollama Embeddings
    |
    v
ChromaDB Collection
```

### Question Answering

```text
Nutzerfrage
    |
    v
Embedding der Frage
    |
    v
Semantisches Retrieval
    |
    v
Top-k Chunks
    |
    v
Prompt Builder
    |
    v
Lokales LLM
    |
    v
RagAnswer mit Quellen
```

### Zusammenfassung

Zusammenfassungen laufen bewusst nicht über normales Top-k-Retrieval. Für vollständige Dokumente wird ein eigener Ablauf genutzt:

1. Alle relevanten Chunks einer Datei laden
2. Chunks in Gruppen zusammenfassen
3. Teilsummaries erzeugen
4. Teilsummaries zu einer finalen Zusammenfassung verdichten
5. Quellen und Anzahl der Teilsummaries ausgeben

## 6. Modulübersicht

- `config.py`: Standardpfade, Modellnamen, Chunk-Parameter und Retrieval-Defaults
- `document_loader.py`: Text- und PDF-Laden, OCR-Optionen, Dokumentmetadaten
- `text_splitter.py`: Chunking mit Überlappung und Boundary-Logik
- `embeddings.py`: Ollama-Embedding-Provider und Fehlerbehandlung
- `vector_store.py`: ChromaDB-Persistenz, Quellenlisten, Chunk-Zugriff
- `retriever.py`: semantisches Retrieval mit optionalem Quellenfilter
- `prompt_builder.py`: RAG-Prompt mit Kontext und Quellenregeln
- `llm_client.py`: Ollama-Client für lokale Generierung
- `rag_pipeline.py`: Verbindung aus Retriever, Prompt und LLM-Antwort
- `summarizer.py`: dokumentweite Map-Reduce-Zusammenfassung
- `evaluation.py`: Retrieval-Evaluation über Beispieltabellen
- `library_store.py`: lokale UI-Konfiguration und gecachte Zusammenfassungen
- `cli.py`: Kommandozeilenoberfläche
- `web_app.py`: lokale HTTP-Browseroberfläche
- `schema.py`: zentrale Datenobjekte

## 7. Daten- und Speicherorte

Repository-nahe Verzeichnisse:

- `data/raw/`: lokale Eingabedokumente, nicht für Git gedacht
- `data/processed/`: generierte Reports, UI-Cache und Ausgaben
- `examples/`: kleine Evaluationsdateien
- `documents/`: Projekt- oder Starterdokumente
- `tests/`: automatisierte Tests

Vector Store:

- Standard: `%TEMP%\local_rag_assistant\vector_store`
- Überschreibbar über `RAG_VECTOR_STORE_DIR`
- Überschreibbar pro Befehl über `--vector-store`

Grund für den temporären Standardpfad:

- Weniger Risiko, große oder binäre ChromaDB-Dateien versehentlich zu committen
- Umgehung möglicher SQLite-I/O-Probleme auf bestimmten Projektlaufwerken
- Einfaches Löschen und Neuaufbauen des Index

## 8. Metadatenanforderungen

Jeder Chunk soll ausreichend Metadaten behalten, damit Antworten überprüfbar bleiben:

- vollständiger Quellpfad
- Dateiname
- Dokumenttyp
- Chunk-Index
- Seitenzahl, falls verfügbar
- Start- und Endzeichen im extrahierten Text, falls verfügbar
- OCR-Hinweis, falls OCR verwendet wurde

Diese Metadaten sind wichtig für:

- Quellenanzeige in Antworten
- Debugging von Retrieval-Ergebnissen
- Quellenfilter in CLI und UI
- Dokumentweite Zusammenfassungen aus indexierten Chunks
- Retrieval-Evaluation

## 9. Prompting-Ansatz

Der RAG-Prompt soll das Modell zu folgenden Verhaltensweisen führen:

- Möglichst nur auf Basis des gelieferten Kontexts antworten
- Fehlenden oder schwachen Kontext transparent benennen
- Quellen referenzieren
- Nicht belegte Details vermeiden
- In der Sprache der Nutzerfrage antworten, wenn möglich

Für Debugging kann der vollständige Prompt mit `--show-prompt` angezeigt werden.

## 10. Retrieval-Ansatz

Das System soll klar zwischen diesen Modi unterscheiden:

- Keyword-Suche: exakte oder oberflächennahe Texttreffer
- Semantisches Retrieval: Embedding-basierte Ähnlichkeit
- Antwortgenerierung: Synthese über gefundene Chunks
- Dokumentzusammenfassung: Verarbeitung aller relevanten Chunks einer Quelle
- Evaluation: Vergleich von Retrieval-Ergebnissen mit erwarteten Quellen oder Evidenzen

Qualitätsanforderungen:

- Retrieval soll nicht nur exakte Stichwörter finden.
- Chunks sollen semantisch sinnvolle Textbereiche enthalten.
- Quellenfilter müssen reproduzierbare Tests und gezielte Analysen ermöglichen.
- Retrieval-Ergebnisse sollen vor Prompt-Tuning direkt inspizierbar sein.

## 11. CLI-Referenz

Die CLI ist für technische Prüfung, Skripting, Evaluation und Debugging gedacht. Die README beschreibt primär die Web UI.

### Web UI starten

```powershell
.\.venv\Scripts\rag-assistant-ui.exe
```

Danach öffnen:

```text
http://127.0.0.1:8765
```

### Dokumente indexieren

Ordner indexieren:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw
```

Ordner mit kleinerem Embedding-Batch indexieren:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw --embedding-batch-size 4
```

Einzelne Datei indexieren:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest path\to\file.pdf --embedding-batch-size 4
```

Chunk-Größe anpassen:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw --chunk-size 900 --chunk-overlap 150
```

OCR für gescannte PDFs aktivieren:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw --ocr --ocr-language eng+deu --ocr-scale 3 --ocr-psm 6 --embedding-batch-size 4
```

Wichtige OCR-Optionen:

- `--ocr-language`: Tesseract-Sprache, zum Beispiel `eng`, `deu` oder `eng+deu`
- `--ocr-scale`: Render-Skalierung vor OCR; `3` oder `4` hilft oft bei kleinem Text
- `--ocr-psm`: Page-Segmentation-Modus; `6` für Textblöcke, `4` für Spalten, `11` für verstreuten Text
- `--no-ocr-preprocess`: Bildvorverarbeitung deaktivieren
- `--no-ocr-clean`: Textbereinigung deaktivieren

### Index inspizieren

Indexierte Quellen anzeigen:

```powershell
.\.venv\Scripts\rag-assistant.exe sources
```

Chunks einer Quelle anzeigen:

```powershell
.\.venv\Scripts\rag-assistant.exe chunks "Lions and Tigers and Snares.pdf" --limit 5 --preview-chars 180
```

### Retrieval und Fragen

Semantische Suche ohne LLM-Antwort:

```powershell
.\.venv\Scripts\rag-assistant.exe retrieve "Welche Methoden werden beschrieben?" --top-k 5
```

Suche auf eine Quelle begrenzen:

```powershell
.\.venv\Scripts\rag-assistant.exe retrieve "Welche Hinweise gibt es?" --source "Lions and Tigers and Snares.pdf"
```

Frage mit lokaler Antwortgenerierung:

```powershell
.\.venv\Scripts\rag-assistant.exe ask "Fasse die Dokumente zusammen." --llm-model qwen3:8b
```

Prompt zur Fehlersuche anzeigen:

```powershell
.\.venv\Scripts\rag-assistant.exe ask "Was sind die wichtigsten Punkte?" --show-prompt
```

Frage auf eine Quelle begrenzen:

```powershell
.\.venv\Scripts\rag-assistant.exe ask "Was steht in diesem Dokument?" --source "Lions and Tigers and Snares.pdf"
```

### Zusammenfassung

Eine Datei direkt zusammenfassen:

```powershell
.\.venv\Scripts\rag-assistant.exe summarize README.md --llm-model qwen3:8b
```

Bereits indexierte Chunks einer Quelle zusammenfassen:

```powershell
.\.venv\Scripts\rag-assistant.exe summarize README.md --from-index --llm-model qwen3:8b
```

Zusammenfassung mit Fokusfrage:

```powershell
.\.venv\Scripts\rag-assistant.exe summarize README.md --question "Implementierungsplan" --max-chunks-per-group 4
```

### Retrieval-Evaluation

Evaluation gegen Beispiel-Fragen ausführen:

```powershell
.\.venv\Scripts\rag-assistant.exe eval examples/retrieval_eval_examples.md --top-k 5
```

Evaluation auf eine Quelle begrenzen:

```powershell
.\.venv\Scripts\rag-assistant.exe eval examples/lions_tigers_retrieval_eval_examples.md --source "Lions and Tigers and Snares.pdf" --top-k 5
```

JSON-Report schreiben:

```powershell
.\.venv\Scripts\rag-assistant.exe eval examples/lions_tigers_retrieval_eval_examples.md --source "Lions and Tigers and Snares.pdf" --top-k 5 --json-report data/processed/lions_tigers_eval_report.json
```

### Vector Store überschreiben

Wenn ChromaDB auf einem Projektlaufwerk einen SQLite-Disk-I/O-Fehler meldet, kann ein Vector-Store-Pfad auf einem anderen lokalen Laufwerk genutzt werden:

```powershell
$env:RAG_VECTOR_STORE_DIR = "$env:TEMP\local_rag_vector_store"
.\.venv\Scripts\rag-assistant.exe ingest data/raw --vector-store $env:RAG_VECTOR_STORE_DIR
```

### Ollama-Modellstatus prüfen

```powershell
ollama ps
```

## 12. Entwicklungsprinzipien

- Kleine, nachvollziehbare Änderungen bevorzugen
- Module klar getrennt halten
- Keine unnötige Framework-Abstraktion hinzufügen
- Bestehende Schnittstellen schonen
- Tests für Kernlogik ergänzen, wenn Verhalten geändert wird
- Große lokale Dateien, Indexdaten, Modelle und Caches nicht committen
- README nutzerorientiert halten
- Technische Details und Entscheidungen in diesem Dokument pflegen

## 13. Teststrategie

Tests sollen Pipeline-Verhalten prüfen, ohne die Intelligenz eines konkreten LLMs bewerten zu müssen.

Wichtige Testbereiche:

- Dokumentladen erzeugt Text und Metadaten
- Chunking verliert keinen Inhalt und respektiert Grenzen
- Chunks enthalten Quelleninformationen
- Vector Store speichert und liefert Chunks
- Retriever akzeptiert Quellenfilter
- Prompt Builder enthält Frage und Kontext
- RAG-Pipeline liefert Antwortobjekt und Quellen
- Summarizer verarbeitet mehrere Chunk-Gruppen
- CLI-Formatter erzeugen lesbare Ausgaben
- Evaluation erkennt passende und fehlende Treffer

Standardtestbefehl:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

## 14. Bekannte Risiken und Grenzen

- PDF-Extraktion ist bei Layouts, Tabellen, Scans und mehrspaltigen Dokumenten fehleranfällig.
- OCR kann langsam sein und fehlerhaften Text erzeugen.
- Lokale Modelle halluzinieren bei schwachem Kontext weiterhin.
- Kleine Modelle folgen Quellen- und Kontextregeln nicht immer zuverlässig.
- Retrieval kann relevante Chunks verpassen, wenn Chunking, Embedding-Modell oder Frageformulierung nicht passen.
- ChromaDB und SQLite können je nach Laufwerk oder Virenscanner I/O-Probleme zeigen.
- Große Dokumentbestände benötigen Evaluation und Tuning statt reinem Bauchgefühl.

## 15. Erfolgskriterien

Eine stabile erste Version ist erreicht, wenn:

- Dokumente lokal indexiert werden können
- Quellen und Chunks inspizierbar sind
- Fragen semantisch relevante Chunks abrufen
- Antworten mit lokalen Modellen generiert werden
- Quellen in Antworten angezeigt werden
- Vollständige Dokumente zusammengefasst werden können
- Tests für Kernlogik laufen
- README und Projektkontext den aktuellen Stand erklären

## 16. Mögliche nächste Schritte

- README um Screenshots der lokalen UI ergänzen
- Architekturdiagramm als Bild oder Mermaid-Diagramm hinzufügen
- Retrieval-Evaluation mit deutschen Dokumenten erweitern
- Hybrid Search mit BM25 und Vektor-Retrieval prüfen
- Optionalen lokalen Reranker integrieren
- Tabellenextraktion verbessern
- UI-Texte vollständig eindeutschen
- Exportformate für Antworten und Zusammenfassungen erweitern
- Kleine Beispiel-Dokumentbibliothek für Portfolio-Demos vorbereiten