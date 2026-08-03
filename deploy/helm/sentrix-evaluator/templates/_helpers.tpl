{{/*
Expand the name of the chart.
*/}}
{{- define "sentrix-evaluator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create chart name and version as a label.
*/}}
{{- define "sentrix-evaluator.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "sentrix-evaluator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "sentrix-evaluator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sentrix-evaluator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Create the chart fullname.
*/}}
{{- define "sentrix-evaluator.fullname" -}}
{{- local := dict "name" (include "sentrix-evaluator.name" .) "instance" .Release.Name -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "sentrix-evaluator.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
