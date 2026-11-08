# PNEUMYST AI design

Exports of the Figma file "PNEUMYST AI - App design", used as the reference while building the Next.js frontend.
All values in the pictures are sample data. Band thresholds in the PDF pages are the real ones from `ml/configs`.

## Screens (desktop, 1440 px)

| Screen | File |
|---|---|
| Landing | [screens/landing.png](screens/landing.png) |
| New scan | [screens/new-scan.png](screens/new-scan.png) |
| Register | [screens/register.png](screens/register.png) |
| History | [screens/history.png](screens/history.png) |
| History, no scans yet | [screens/history-empty.png](screens/history-empty.png) |
| History, account menu open | [screens/history-account-menu.png](screens/history-account-menu.png) |
| Results, Report tab | [screens/results-report.png](screens/results-report.png) |
| Results, summary being written | [screens/results-writing-summary.png](screens/results-writing-summary.png) |
| Results, Ask tab | [screens/results-ask.png](screens/results-ask.png) |
| Results, Details tab | [screens/results-details.png](screens/results-details.png) |
| Results, overlay viewer | [screens/results-overlay.png](screens/results-overlay.png) |

Still to export: Screening in progress and Sign in.

## Phone (390 px)

| Screen | File |
|---|---|
| New scan | [phone/new-scan.png](phone/new-scan.png) |
| Results | [phone/results.png](phone/results.png) |
| Menu open | [phone/menu-open.png](phone/menu-open.png) |

## Documents (A4)

| Document | File |
|---|---|
| Screening Analysis Report, pages 1 to 3 | [pdf/report-page-1.png](pdf/report-page-1.png), [pdf/report-page-2.png](pdf/report-page-2.png), [pdf/report-page-3.png](pdf/report-page-3.png) |
| Printable audit record | [pdf/audit-record.png](pdf/audit-record.png) |

The report pages restyle what `backend/app/reporting/pdf.py` already writes. The audit page is a printable view of
`audit.json` and needs a new endpoint before it exists.

## Boards and assets

- [boards/error-states.png](boards/error-states.png): every error the backend returns, with the wording the user sees.
- [boards/components.png](boards/components.png): logo, buttons, fields, nav item, footer link and social link.
- [assets/lung-artwork.png](assets/lung-artwork.png): the lung artwork used on the landing page.

## Design tokens

| Token | Value | Use |
|---|---|---|
| brand/green | `#16A34A` | buttons, glass nav, headings, links |
| brand/green-deep | `#14532D` | small text on mint, active pill text |
| brand/mint | `#ECFDF3` | soft backgrounds, selected rows |
| brand/mint-strong | `#BBF7D0` | borders on mint, hover states |
| surface/page | `#F7FBF8` | page background |
| surface/card | `#FFFFFF` | cards and panels |
| surface/viewer | `#0B1220` | dark X-ray viewer |
| border/default | `#E2E8F0` | card and field borders |
| text/primary | `#0F172A` | body text |
| text/muted | `#64748B` | secondary text |
| band/likely | `#B3261E` | likely findings only |
| band/possible | `#9A5B00` | possible findings and warnings |
| band/unlikely | `#4D4D4D` | unlikely findings |
| status/caution-bg | `#FFF4D6` | caution and warning boxes |

Green is never used for a finding band, so a result can never read as "all clear".

Fonts: Playfair Display Bold Italic for headings, Inter for body text, JetBrains Mono for small uppercase labels and
buttons, Montserrat ExtraBold for the logo wordmark.

Glass effects (nav bar, active nav pill, Send button) are described in the Figma file; transparent colours there are
set as plain hex values because variable-bound colours lost their opacity.
