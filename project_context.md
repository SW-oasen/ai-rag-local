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
- OCR ist rechenintensiv und wird in der UI standardmäßig angeboten, aber nur für PDF-Seiten ohne extrahierbaren Text genutzt.
- Der Vector Store liegt standardmäßig im Projektordner unter `vector_store`.

## 3. Aktueller Implementierungsstand

Vorhanden:

- Dokumentladen für Markdown, Text, PDFs, EPUB, AZW3 und OpenDocument-Dateien
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
- Fortschrittsmeldungen für dokumentweite Zusammenfassungen
- Markdown-Rendering in der Web UI für Antworten und Zusammenfassungen
- Retrieval-Evaluation über Markdown-Beispieldateien
- CLI mit `ingest`, `sources`, `delete-source`, `reset-index`, `chunks`, `retrieve`, `ask`, `summarize`, `profiles` und `eval`
- Lokale Browser-UI mit `Overview`, `Ask`, `Summarize`, `Extract Text` und `Configuration`
- Indexverwaltung zum Löschen einzelner Quellen und Zurücksetzen des Vector Store
- Statusbasierte `Configured Paths` in der Web UI
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

Optionale E-Book-Technologien:

- Calibre `ebook-convert` für AZW3-Textextraktion

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

Die CLI und Web UI können dabei Fortschrittsmeldungen ausgeben:

- Anzahl der Chunks und Gruppen
- Start und Ende jeder Gruppe
- Dauer pro Gruppe
- Start und Ende des finalen Merge-Schritts

## 6. Modulübersicht

- `config.py`: Standardpfade, Modellnamen, Chunk-Parameter und Retrieval-Defaults
- `document_loader.py`: Text-, PDF-, EPUB-, AZW3- und OpenDocument-Laden, OCR-Optionen, Dokumentmetadaten
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
- `profile_store.py`: JSON-basierte RAG-Profile mit `general`-Default
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

- Standard: `vector_store` im Projektordner
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
- Antworten als Markdown strukturieren, zum Beispiel mit Überschriften, Listen, nummerierten Schritten und Fettschrift

Für Debugging kann der vollständige Prompt mit `--show-prompt` angezeigt werden.

Die Web UI rendert einen bewusst kleinen, sicheren Markdown-Subset:

- Überschriften `#` bis `######`
- ungeordnete und nummerierte Listen
- verschachtelte Bulletpoints unter nummerierten Punkten
- Fettschrift mit `**...**`
- Codeblöcke
- gruppenartige Zeilen mit Doppelpunkt als Überschrift mit eingerückten Unterpunkten

HTML wird dabei escaped; die UI rendert keine ungeprüfte Roh-HTML-Ausgabe.

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

Profile anzeigen:

```powershell
.\.venv\Scripts\rag-assistant.exe profiles
```

Ordner indexieren:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw
```

Ordner mit Profil indexieren:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw/local-docus/tech --profile technical
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

- `--ocr-language`: Tesseract-Sprache, zum Beispiel `eng`, `deu`, `eng+deu`, `fra`, `chi_sim` oder `chi_tra`
- `--ocr-scale`: Render-Skalierung vor OCR; `3` oder `4` hilft oft bei kleinem Text
- `--ocr-psm`: Page-Segmentation-Modus; `6` für Textblöcke, `4` für Spalten, `11` für verstreuten Text
- `--no-ocr-preprocess`: Bildvorverarbeitung deaktivieren
- `--no-ocr-clean`: Textbereinigung deaktivieren

Die Web UI bietet für OCR-Sprachen eine Dropdownliste mit Englisch, Deutsch, Englisch+Deutsch, Französisch sowie vereinfachtem und traditionellem Chinesisch. Die jeweilige Sprache funktioniert nur, wenn das passende Tesseract-Sprachpaket lokal installiert ist. Bei vollständig leerer Textextraktion zeigt die UI einen Hinweis auf OCR und Tesseract-Sprachpakete. `.txt`-Exporte aus `Extract Text` verwenden einen Dateinamen nach dem Muster `<quelle>-extracted.txt`.

### Index inspizieren

Indexierte Quellen anzeigen:

```powershell
.\.venv\Scripts\rag-assistant.exe sources
```

Chunks einer Quelle anzeigen:

```powershell
.\.venv\Scripts\rag-assistant.exe chunks "Lions and Tigers and Snares.pdf" --limit 5 --preview-chars 180
```

Einzelne Quelle aus dem Index löschen:

```powershell
.\.venv\Scripts\rag-assistant.exe delete-source "Lions and Tigers and Snares.pdf"
```

Gesamten Index zurücksetzen:

```powershell
.\.venv\Scripts\rag-assistant.exe reset-index --yes
```

### Retrieval und Fragen

Semantische Suche ohne LLM-Antwort:

```powershell
.\.venv\Scripts\rag-assistant.exe retrieve "Welche Methoden werden beschrieben?" --top-k 5
```

Semantische Suche innerhalb eines Profils:

```powershell
.\.venv\Scripts\rag-assistant.exe retrieve "Welche Methoden werden beschrieben?" --profile technical --top-k 5
```

Suche auf eine Quelle begrenzen:

```powershell
.\.venv\Scripts\rag-assistant.exe retrieve "Welche Hinweise gibt es?" --source "Lions and Tigers and Snares.pdf"
```

Frage mit lokaler Antwortgenerierung:

```powershell
.\.venv\Scripts\rag-assistant.exe ask "Fasse die Dokumente zusammen." --llm-model qwen3:8b
```

Frage mit Profilfilter:

```powershell
.\.venv\Scripts\rag-assistant.exe ask "Wie nutze ich die API?" --profile technical --llm-model qwen3:8b
```

Die Antwort wird als Markdown angefordert. In der Web UI wird ein sicherer Markdown-Subset formatiert dargestellt; in der CLI bleibt die Markdown-Ausgabe als Text sichtbar.

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

Während der Zusammenfassung werden Fortschrittsmeldungen ausgegeben, zum Beispiel Gruppenfortschritt und Dauer des finalen Merge-Schritts. Das ist besonders bei großen PDFs wichtig, weil einzelne LLM-Aufrufe mehrere Minuten dauern können.

### Web-UI-Konfiguration

`Configured Paths` zeigt Pfade mit Status und passenden Aktionen:

- `Not indexed`: keine passende Quelle im Vector Store
- `Indexed`: eine konfigurierte Einzeldatei ist indexiert
- `Indexed folder`: alle unterstützten Dateien im Ordner sind indexiert
- `Partially indexed folder`: nur ein Teil der unterstützten Dateien im Ordner ist indexiert
- `Contains ... indexed sources`: ein nicht eindeutig prüfbarer Ordner/Pfad enthält indexierte Quellen

Aktionen:

- `Ingest`: nicht indexierten Pfad indexieren
- `Ingest Missing`: fehlende Quellen in einem teilweise indexierten Ordner nachziehen
- `Re-ingest Folder`: Ordner erneut indexieren
- `Delete Index`: indexierte Einzeldatei aus dem Vector Store löschen
- `Remove Path`: Pfad nur aus der UI-Konfiguration entfernen

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
$env:RAG_VECTOR_STORE_DIR = "$env:LOCALAPPDATA\local_rag_assistant\vector_store"
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
- Web UI rendert Markdown-Ausgaben sicher
- Statuslogik der konfigurierten Pfade berücksichtigt Indexzustand
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
- Profil-System für szenariospezifische RAG-Modi planen und einführen

## 17. Profile

Das aktuelle RAG-System ist bewusst generisch. Für bessere Qualität in konkreten Szenarien werden Profile eingeführt. Ein Profil definiert, welche Quellen genutzt werden und wie Chunking, Retrieval, Prompting und Antwortformatierung aussehen.

Beispiele:

- `general`: allgemeine Dokumentfragen
- `technical`: technische Dokumentation, APIs, Installationsschritte
- `recipes`: Kochbücher, Zutaten, Kategorien und Zubereitung
- `research`: wissenschaftliche PDFs mit Methoden, Ergebnissen und Limitationen
- `legal`: vorsichtige, quellennahe Antworten für Verträge oder Regeltexte

Eine Datei darf mehreren Profilen zugeordnet sein. Die Profilzugehörigkeit wird als Metadatum an Chunks geführt. Für Nicht-`general`-Profile wird das Profil in die Chunk-ID einbezogen, damit dieselbe Quelle parallel in mehreren Profilen indexiert werden kann.

Bereits umgesetzt:

- `ProfileStore` mit JSON-Speicherung unter `data/processed/profiles.json`
- Default-Profil `general`
- automatisches Anlegen neuer Profile bei `ingest --profile <name>`
- CLI-Befehl `profiles`
- `--profile` für `ingest`, `retrieve` und `ask`
- Profil-Metadaten beim Ingest
- Profilfilter im semantischen Retrieval
- Prompt-Stile für `general`, `technical`, `recipes`, `research` und `legal`
- automatische Prompt-Stil-Auswahl für neue Profile mit bekannten Namen
- Profil-Dropdown im Web-UI-Bereich `Ask`
- Web-UI `Retrieve` und `Ask` nutzen Profilfilter und passenden Prompt-Stil
- Profilverwaltung in der Web UI unter `Configuration`
- Pfade pro Profil in der Web UI hinzufügen und entfernen
- Profilpfade aus der Web UI direkt mit Profilmetadaten indexieren
- bestehende Chunks ohne Profil-Metadatum werden beim Retrieval als `general` behandelt
- Profilpfade zeigen in der Web UI ihren Indexstatus pro Profil

Nächste Etappen:

1. Profil-UX nach praktischer Nutzung weiter glätten, zum Beispiel Profilbeschreibung und Chunk-Parameter direkt editierbar machen.
