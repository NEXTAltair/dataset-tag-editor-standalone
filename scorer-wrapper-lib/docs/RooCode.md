## RooCode拡張機能の設定

RooCode拡張機能は、以下の主要な設定を詳細にサポートしています:

- **モード設定**  
  - 各モード (Code、Architect、Ask) に対し、プロンプトのカスタマイズや専用の挙動設定が可能です。  
  - モード間のシームレスな切り替えや、初期設定の自動ロード機能も提供します。

- **API設定**  
  - 利用するAPIに応じたエンドポイント、認証キー、タイムアウト設定などの細かなカスタマイズが行えます。  
  - OpenAI、Anthropic など、複数のプロバイダーに対応し、モードごとに異なるAPI設定を適用可能です。

- **ファイル＆エディタ操作**  
  - ファイルの作成、編集、リネーム、削除などの自動操作を、ユーザーのワークフローやルールに合わせて構成できます。  
  - 編集許可されたファイルタイプやディレクトリの制限を設け、操作の安全性を担保します。

- **承認オプション**  
  - 自動、手動、およびハイブリッドの承認方式をサポートし、各タスク実行前にユーザーによる確認を促す設定が可能です。  
  - タスクごとのリスク度や実行環境に合わせた柔軟な承認プロセスを構築します。

## 詳細設定項目

### Provider Settings & Configuration Profile
- 複数のAPI構成をローカルに保存し、迅速な切り替えが可能です。  
- API Provider: APIリクエスト用の認証キーがローカルに保存されます。  
- Model: 画像対応、最大トークン数、出力のランダム性などのパラメータが設定できます。  
> ※ 詳細解説: このセクションでは、各APIプロバイダーとの接続情報や利用パラメータ（認証キー、エンドポイント、タイムアウト設定など）の管理方法について説明します。ユーザーはここで、自身のワークフローに合わせた最適なAPI設定を行い、プロファイルとして保存できます。

### Auto-Approve Settings
- 自動承認機能を有効にすると、ディレクトリ内容の自動閲覧やファイルの自動作成・編集が実行されます。  
  > ※ 詳細: この設定により、変更が必要なファイルの自動作成や編集を、ユーザーの手動承認なしで実行します。開発スピードは向上しますが、予期しない変更リスクも含みます。
- ブラウザ操作、失敗したAPIリクエストの自動リトライ、モード切替や新規タスク作成の自動実行も含みます。  
  > ※ 詳細: 自動ブラウザ操作はUIテストや自動操作に役立ち、リトライ機能は一時的なエラーを自動で回復します。モード切替により、最適な動作環境を維持できます。
- 実行前の遅延時間や、各MCPツールの自動承認設定も調整可能です。  
  > ※ 詳細: 遅延時間の設定は、不測の事態に備えるためのクールダウン期間として動作し、個々のMCPツールごとに柔軟な自動承認ルールを提供します。
- 自動モード切替  
  > ※ 詳細: 状況に応じたAIモードの自動切替を実現し、ユーザー操作を最小限に抑え、シームレスな作業フローをサポートします。
- 自動端末コマンド実行（Allowed Auto-Execute Commands）  
  > ※ 詳細: 予め許可されたコマンド一覧に基づいて、ターミナルでの操作を自動実行し、手動承認の手間を省きます。

### Browser Settings
- ビューポートサイズの選択や、スクリーンショット品質を調整できます。  
- WebP品質の設定により、画質とトークン使用量のバランスを管理します。  
> ※ 詳細解説: ここでは、ブラウザ操作に関する設定項目を取り扱います。ユーザーは、リモートブラウザ操作時の表示サイズや画面キャプチャの品質を調整することで、より正確な動作検証や効率的なデバッグ環境を構築できます。

### Notification & Advanced Settings
- 通知時にサウンド効果を再生できるオプションがあります。  
- APIリクエストのレート制限や、ターミナル出力行数の上限を設定でき、不要な出力を削減します。  
- ファイルの変更時に自動でチェックポイントを保存したり、実験的な検索置換・挿入ツールを有効化することが可能です。  
> ※ 詳細解説: このセクションでは、通知の挙動および高度な設定について説明します。APIのリクエスト頻度制御、ターミナルの出力制限、ファイル操作時の自動バックアップなど、システムの安定性や効率を保証するための詳細な設定項目を扱います。

## プロンプト設定項目

Promptsは、Rooの振る舞いやユーザーとの対話の基盤となる設定です。以下の項目は、それぞれの役割と設定例を示しています。

### Preferred Language
- 役割: Rooが対話に使用する言語を指定し、ユーザーとの一貫したコミュニケーションを実現します。
- 設定例: 「Japanese - 日本語」と指定することで、全ての返答が日本語となります。

### Custom Instructions for All Modes
- 役割: 全モード共通の基本行動指針を定義し、Rooの基本的な動作パターンを決定します。
- 設定例: 
  - 返答は日本語で行うこと  
  - DocstringsはGoogleスタイルで記述  
  - テストコードを書きやすい実装にする  
  - 更新前のコードを誤って消さないこと

### Mode-Specific Prompts
- 役割: 各専用モード（Code、Architect、Ask）に対し、動作指針や役割を個別に設定します。
- 設定例:
  - Code: コーディングタスクに特化。例：「コードの修正案を提示」、「効率的な実装例を提供」
  - Architect: システム設計に最適化。例：「全体アーキテクチャの設計案を説明」、「システム構成を図示」
  - Ask: 質問対応に注力。例：「技術的な疑問に分かりやすく回答」、「追加情報の確認を促す」

### Role Definition
- 役割: 各モードにおけるRooの専門性や人格を明確にし、振る舞いの基準を設定します。
- 設定例: 「You are Roo, a knowledgeable technical assistant focused on answering questions and providing information about software development, technology, and related topics.」

### API Configuration & Available Tools
- 役割: プロンプトの実行に必要なAPI設定や利用可能なツールの指定を行います。
- 設定例: 各モード毎に、使用するAPIプロファイルの選択や、Read Files, Edit Files（Markdownのみ）、Use Browser, Use MCP などのツール利用権限を付与します。

### Additional Prompt Functions
- 役割: プロンプトを拡張し、入力に応じた改善提案や支援を自動で行います。
- 設定例: 
  - Enhance Prompt: ユーザー入力の意図に沿った改良版プロンプトを生成  
  - Explain Code: コードの説明を自動出力  
  - Fix Issues: 問題点の修正案を提示  
  - Add to Context: 必要な情報を適宜プロンプトに追加する