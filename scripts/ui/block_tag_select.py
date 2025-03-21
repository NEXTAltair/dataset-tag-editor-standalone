from typing import Callable
import gradio as gr

from .ui_common import *

TagFilter = dte_module.filters.TagFilter
Filter = dte_module.filters.Filter

SortBy = dte_instance.SortBy
SortOrder = dte_instance.SortOrder


class TagSelectUI:
    def __init__(self):
        self.filter_word = ""
        self.sort_by = SortBy.ALPHA
        self.sort_order = SortOrder.ASC
        self.selected_tags = set()
        self.tags = set()
        self.get_filters = lambda: []
        self.prefix = False
        self.suffix = False
        self.regex = False

    def create_ui(
        self,
        get_filters: Callable[[], list[Filter]],
        sort_by=SortBy.ALPHA,
        sort_order=SortOrder.ASC,
        prefix=False,
        suffix=False,
        regex=False,
    ):
        self.get_filters = get_filters
        self.prefix = prefix
        self.suffix = suffix
        self.regex = regex

        self.tb_search_tags = gr.Textbox(label="Search Tags", interactive=True)
        with gr.Row():
            self.cb_prefix = gr.Checkbox(label="Prefix", value=False, interactive=True)
            self.cb_suffix = gr.Checkbox(label="Suffix", value=False, interactive=True)
            self.cb_regex = gr.Checkbox(
                label="Use regex", value=False, interactive=True
            )
        with gr.Row():
            self.rb_sort_by = gr.Radio(
                choices=[e.value for e in SortBy],
                value=sort_by,
                interactive=True,
                label="Sort by",
            )
            self.rb_sort_order = gr.Radio(
                choices=[e.value for e in SortOrder],
                value=sort_order,
                interactive=True,
                label="Sort Order",
            )
        with gr.Row():
            self.btn_select_visibles = gr.Button(value="Select visible tags")
            self.btn_deselect_visibles = gr.Button(value="Deselect visible tags")
        self.cbg_tags = gr.CheckboxGroup(label="Select Tags", interactive=True)

    def set_callbacks(self):
        o_update = self.cbg_tags
        self.tb_search_tags.change(
            fn=self.tb_search_tags_changed,
            inputs=self.tb_search_tags
        ).then(
            fn=self.cbg_tags_update,
            outputs=o_update
        )
        self.cb_prefix.change(
            fn=self.cb_prefix_changed,
            inputs=self.cb_prefix
        ).then(
            fn=self.cbg_tags_update,
            outputs=o_update
        )
        self.cb_suffix.change(
            fn=self.cb_suffix_changed,
            inputs=self.cb_suffix
        ).then(
            fn=self.cbg_tags_update,
            outputs=o_update
        )
        self.cb_regex.change(
            fn=self.cb_regex_changed,
            inputs=self.cb_regex
        ).then(
            fn=self.cbg_tags_update,
            outputs=o_update
        )
        self.rb_sort_by.change(
            fn=self.rd_sort_by_changed,
            inputs=self.rb_sort_by
        ).then(
            fn=self.cbg_tags_update,
            outputs=o_update
        )
        self.rb_sort_order.change(
            fn=self.rd_sort_order_changed,
            inputs=self.rb_sort_order
        ).then(
            fn=self.cbg_tags_update,
            outputs=o_update
        )
        self.btn_select_visibles.click(
            fn=self.btn_select_visibles_clicked, outputs=o_update
        )
        self.btn_deselect_visibles.click(
            fn=self.btn_deselect_visibles_clicked,
            inputs=self.cbg_tags,
            outputs=o_update,
        )
        self.cbg_tags.change(
            fn=self.cbg_tags_changed, inputs=self.cbg_tags, outputs=o_update
        )

    def tb_search_tags_changed(self, tb_search_tags: str):
        self.filter_word = tb_search_tags

    def cb_prefix_changed(self, prefix: bool):
        self.prefix = prefix

    def cb_suffix_changed(self, suffix: bool):
        self.suffix = suffix

    def cb_regex_changed(self, regex: bool):
        self.regex = regex

    def rd_sort_by_changed(self, rb_sort_by: str):
        self.sort_by = rb_sort_by

    def rd_sort_order_changed(self, rd_sort_order: str):
        self.sort_order = rd_sort_order

    def cbg_tags_changed(self, cbg_tags: list[str]):
        self.selected_tags = set(dte_instance.read_tags(cbg_tags))
        return self.cbg_tags_update()

    def btn_deselect_visibles_clicked(self, cbg_tags: list[str]):
        tags = dte_instance.get_filtered_tags(
            self.get_filters(), self.filter_word, True
        )
        selected_tags = set(dte_instance.read_tags(cbg_tags)) & tags
        self.selected_tags -= selected_tags
        return self.cbg_tags_update()

    def btn_select_visibles_clicked(self):
        tags = set(
            dte_instance.get_filtered_tags(self.get_filters(), self.filter_word, True)
        )
        self.selected_tags |= tags
        return self.cbg_tags_update()

    def cbg_tags_update(self):
        tags = dte_instance.get_filtered_tags(
            self.get_filters(),
            self.filter_word,
            True,
            prefix=self.prefix,
            suffix=self.suffix,
            regex=self.regex,
        )
        self.tags = set(
            dte_instance.get_filtered_tags(
                self.get_filters(),
                filter_tags=True,
                prefix=self.prefix,
                suffix=self.suffix,
                regex=self.regex,
            )
        )
        self.selected_tags &= self.tags
        tags = dte_instance.sort_tags(
            tags=tags, sort_by=self.sort_by, sort_order=self.sort_order
        )
        tags = dte_instance.write_tags(tags, self.sort_by)
        selected_tags = dte_instance.write_tags(list(self.selected_tags), self.sort_by)
        return gr.CheckboxGroup(value=selected_tags, choices=tags)
