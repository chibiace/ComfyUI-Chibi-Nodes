import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "chibi_nodes",

  beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "Textbox") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        return r;
      };

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);

        for (const widget of this.widgets) {
          if (widget.type === "customtext") {
            widget.value = message.text.join("");
          }
        }

        this.onResize?.(this.size);
      };
    }
    if (nodeData.name === "ImageSizeInfo") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        const iw = ComfyWidgets["INT"](this, "width", ["INT", {}], app).widget;
        const ih = ComfyWidgets["INT"](this, "height", ["INT", {}], app).widget;
        return r;
      };

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);

        for (const widget of this.widgets) {
          // console.log(widget.type);
          // console.log(widget.name);
          if (widget.name == "width") {
            console.log(message);
            widget.value = message.width[0];
          }
          if (widget.name == "height") {
            console.log(message);
            widget.value = message.height[0];
          }
        }

        this.onResize?.(this.size);
      };
    }
  },
});
